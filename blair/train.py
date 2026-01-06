import logging
import os
import sys
from dataclasses import dataclass, field
from io import BytesIO
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast

import torch

from PIL import Image
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
    BertModel,
    BertForPreTraining,
    CLIPImageProcessor,
    RobertaModel,
)
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process
# from transformers.utils import cached_property, is_torch_available
from multimodal.blair_clip import BlairCLIPDualEncoder
from simcse.models import RobertaForCL, BertForCL
from simcse.trainers import CLTrainer

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # SimCSE's arguments
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    ) 
    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )
    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    mlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )

    model_family: str = field(
        default="text",
        metadata={
            "help": "Select `text` for the original SimCSE/BLaIR objective or `blair_clip` for the multimodal twin tower."
        },
    )
    mm_clip_model_name: str = field(
        default="openai/clip-vit-base-patch16",
        metadata={"help": "CLIP vision backbone to use when `model_family` is multimodal."},
    )
    mm_projection_dim: int = field(
        default=512,
        metadata={"help": "Shared embedding dimension used by the multimodal projection heads."},
    )
    mm_temperature_init: float = field(
        default=0.07,
        metadata={"help": "Initial temperature (tau) for the CLIP-style contrastive loss."},
    )
    mm_text_text_weight: float = field(
        default=0.0,
        metadata={"help": "Optional weight for retaining the original text-text contrastive loss."},
    )
    mm_freeze_clip_tower: bool = field(
        default=False,
        metadata={"help": "Freeze the CLIP vision encoder during multimodal training."},
    )
    mm_freeze_text_tower: bool = field(
        default=False,
        metadata={"help": "Freeze the BLaIR text encoder during multimodal training."},
    )

    def is_multimodal(self) -> bool:
        return (self.model_family or "text").lower() in {"multimodal", "blair_clip", "blairmm"}


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # SimCSE's arguments
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "Optional evaluation data file (.txt, .csv, .tsv, or .json)."},
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    mlm_probability: float = field(
        default=0.15, 
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )
    image_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "Column containing local image paths or PIL-compatible payloads for multimodal training."
        },
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Optional base directory that is prepended to relative entries from `image_column`."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "tsv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "tsv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


@dataclass
class OurTrainingArguments(TrainingArguments):
    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate 
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    is_multimodal = model_args.is_multimodal()
    image_feature_key = "image_path"

    if is_multimodal and data_args.image_column is None:
        raise ValueError(
            "Multimodal training requires `--image_column` that points to a column containing image paths."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    extension = data_args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    if extension in ["csv", "tsv"]:
        datasets = load_dataset('csv', data_files=data_files, cache_dir="./data/", delimiter="\t" if "tsv" in data_args.train_file else ",", lineterminator='\n', on_bad_lines='skip')
        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                'csv', data_files=data_files, cache_dir="./data/", delimiter="\t" if "tsv" in data_args.train_file else ",",
                split=f"train[:{data_args.validation_split_percentage}%]", lineterminator='\n', on_bad_lines='skip'
            )
            datasets["train"] = load_dataset(
                'csv', data_files=data_files, cache_dir="./data/", delimiter="\t" if "tsv" in data_args.train_file else ",",
                split=f"train[{data_args.validation_split_percentage}%:]", lineterminator='\n', on_bad_lines='skip'
            )
    else:
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/")

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if not model_args.model_name_or_path:
        raise NotImplementedError

    if is_multimodal:
        if 'roberta' in model_args.model_name_or_path:
            text_encoder = RobertaModel.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                add_pooling_layer=False,
            )
        elif 'bert' in model_args.model_name_or_path:
            text_encoder = BertModel.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                add_pooling_layer=False,
            )
        else:
            raise NotImplementedError

        model = BlairCLIPDualEncoder(
            text_encoder=text_encoder,
            pooler_type=model_args.pooler_type,
            projection_dim=model_args.mm_projection_dim,
            clip_model_name=model_args.mm_clip_model_name,
            logit_scale_init=model_args.mm_temperature_init,
            text_temp=model_args.temp,
            text_text_weight=model_args.mm_text_text_weight,
            freeze_text=model_args.mm_freeze_text_tower,
            freeze_vision=model_args.mm_freeze_clip_tower,
            mlp_only_train=model_args.mlp_only_train,
            do_mlm=model_args.do_mlm,
            mlm_weight=model_args.mlm_weight,
            cache_dir=model_args.cache_dir,
            model_args=model_args,
        )

        if model_args.do_mlm:
            pretrained_mlm = AutoModelForMaskedLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            source_head = getattr(pretrained_mlm, "lm_head", None)
            if source_head is None and hasattr(pretrained_mlm, "cls"):
                source_head = getattr(pretrained_mlm.cls, "predictions", None)
            if source_head is not None and hasattr(model, "lm_head"):
                model.lm_head.load_state_dict(source_head.state_dict())
    else:
        if 'roberta' in model_args.model_name_or_path:
            model = RobertaForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args
            )
        elif 'bert' in model_args.model_name_or_path:
            model = BertForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args
            )
            if model_args.do_mlm:
                pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
                model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
        else:
            raise NotImplementedError
        # logger.info("Training new model from scratch")
        # model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Prepare features
    column_names = datasets["train"].column_names
    text_column_names = column_names
    if is_multimodal and data_args.image_column in column_names:
        text_column_names = [col for col in column_names if col != data_args.image_column]

    sent2_cname = None
    if len(text_column_names) == 2:
        # Pair datasets
        sent0_cname = text_column_names[0]
        sent1_cname = text_column_names[1]
    elif len(text_column_names) == 3:
        # Pair datasets with hard negatives
        sent0_cname = text_column_names[0]
        sent1_cname = text_column_names[1]
        sent2_cname = text_column_names[2]
    elif len(text_column_names) == 1:
        # Unsupervised datasets
        sent0_cname = text_column_names[0]
        sent1_cname = text_column_names[0]
    else:
        raise NotImplementedError

    def prepare_features(examples):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the 
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.
        total = len(examples[sent0_cname])

        # Avoid "None" fields 
        for idx in range(total):
            if examples[sent0_cname][idx] is None:
                examples[sent0_cname][idx] = " "
            if examples[sent1_cname][idx] is None:
                examples[sent1_cname][idx] = " "
        
        sentences = examples[sent0_cname] + examples[sent1_cname]

        # If hard negative exists
        if sent2_cname is not None:
            for idx in range(total):
                if examples[sent2_cname][idx] is None:
                    examples[sent2_cname][idx] = " "
            sentences += examples[sent2_cname]

        sent_features = tokenizer(
            sentences,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        features = {}
        if sent2_cname is not None:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
        else:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
        
        if is_multimodal:
            image_values = examples.get(data_args.image_column, [None] * total)
            processed = []
            for value in image_values:
                processed.append(value)
            features[image_feature_key] = processed

        return features

    train_dataset = None
    eval_dataset = None

    if training_args.do_train:
        train_dataset = datasets["train"].map(
            prepare_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        eval_dataset = datasets['validation'].map(
            prepare_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    image_processor = None
    if is_multimodal:
        image_processor = CLIPImageProcessor.from_pretrained(
            model_args.mm_clip_model_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # Data collator
    @dataclass
    class OurDataCollatorWithPadding:

        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        mlm: bool = True
        mlm_probability: float = data_args.mlm_probability
        image_key: Optional[str] = None
        image_processor: Optional[CLIPImageProcessor] = None
        image_root: Optional[str] = None

        def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
            bs = len(features)
            if bs == 0:
                return {}

            num_sent = len(features[0]['input_ids'])
            image_values = [feature.get(self.image_key) if self.image_key else None for feature in features]

            flat_features = []
            for feature in features:
                for i in range(num_sent):
                    entry = {}
                    for key, value in feature.items():
                        if key == self.image_key:
                            continue
                        entry[key] = value[i] if key in special_keys else value
                    flat_features.append(entry)

            batch = self.tokenizer.pad(
                flat_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            if model_args.do_mlm:
                batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])

            batch = {
                k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0]
                for k in batch
            }

            if self.image_key and self.image_processor is not None:
                batch["pixel_values"] = self._build_image_batch(image_values)

            if "label" in batch:
                batch["labels"] = batch["label"]
                del batch["label"]
            if "label_ids" in batch:
                batch["labels"] = batch["label_ids"]
                del batch["label_ids"]

            return batch
        
        def mask_tokens(
            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
            """
            inputs = inputs.clone()
            labels = inputs.clone()
            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                ]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            else:
                special_tokens_mask = special_tokens_mask.bool()

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # We only compute loss on masked tokens

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            return inputs, labels

        def _build_image_batch(self, image_values: Sequence[Optional[object]]) -> torch.Tensor:
            if self.image_processor is None:
                raise ValueError("image_processor must be provided when image_key is set.")
            images = [self._load_image(value) for value in image_values]
            pixel_batch = self.image_processor(images=images, return_tensors="pt")
            return pixel_batch["pixel_values"]

        def _load_image(self, value: Optional[object]) -> Image.Image:
            if value is None:
                return self._blank_image()

            if isinstance(value, Image.Image):
                return cast(Image.Image, value).convert("RGB")

            path = str(value)
            if path.startswith("http"):
                try:
                    import requests  # Local import to avoid hard dependency if unused
                    response = requests.get(path, timeout=5)
                    response.raise_for_status()
                    return Image.open(BytesIO(response.content)).convert("RGB")
                except Exception:
                    return self._blank_image()

            resolved = path
            if self.image_root and not os.path.isabs(resolved):
                resolved = os.path.join(self.image_root, resolved)
            try:
                with Image.open(resolved) as img:
                    return img.convert("RGB")
            except Exception:
                return self._blank_image()

        def _blank_image(self) -> Image.Image:
            width, height = self._default_image_size()
            return Image.new("RGB", (width, height), color=0)

        def _default_image_size(self) -> Tuple[int, int]:
            if self.image_processor is None:
                return (224, 224)
            size = getattr(self.image_processor, "size", 224)

            def _to_int(value: object, fallback: int) -> int:
                if isinstance(value, int):
                    return value
                if isinstance(value, float):
                    return int(value)
                try:
                    return int(value)  # type: ignore[arg-type]
                except Exception:
                    return fallback

            if isinstance(size, dict):
                width_val = size.get("width") or size.get("longest_edge") or size.get("shortest_edge") or 224
                height_val = size.get("height") or size.get("shortest_edge") or width_val
                return (_to_int(width_val, 224), _to_int(height_val, 224))
            if isinstance(size, (list, tuple)) and len(size) == 2:
                width_val = _to_int(size[0], 224)
                height_val = _to_int(size[1], width_val)
                return (width_val, height_val)
            scalar = _to_int(size, 224)
            return (scalar, scalar)

    if data_args.pad_to_max_length and not is_multimodal:
        data_collator = default_data_collator
    else:
        data_collator = OurDataCollatorWithPadding(
            tokenizer,
            padding="max_length" if data_args.pad_to_max_length else True,
            image_key=image_feature_key if is_multimodal else None,
            image_processor=image_processor if is_multimodal else None,
            image_root=data_args.image_root,
        )

    # safetensors cannot handle the shared projection weights used inside the multimodal encoder.
    training_args.save_safetensors = False

    trainer = CLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        model_args=model_args,
    )
    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        train_result = trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate(eval_senteval_transfer=True)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results


if __name__ == "__main__":
    main()
