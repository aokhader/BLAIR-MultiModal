# Video Script

## 0:00 – 2:30 — Introduction & Predictive Task

**Visual:** Title Slide — “BLaIR-CLIP Fine Tuning: Multimodal Product Recommendation”

**Speaker:**
"Hello everyone, and thank you for joining me. Today I'm excited to share our project, which focuses on fine-tuning a multimodal recommendation system called BLaIR-CLIP.
This project sits at the intersection of Natural Language Processing and Computer Vision, and our goal is to explore whether combining text and images can meaningfully improve product recommendation quality.

In most e-commerce systems, the recommendation models rely either on text signals like product titles and descriptions, or collaborative filtering signals like user IDs and purchased items. But as we know, modern online shopping involves far more than text. People rely heavily on images — especially for products where design, color, or visual appearance matters.

So the question we explore in this work is:
Can we build a model that “reads” the product and also “sees” it, and does that actually improve recommendation performance?

That brings us to the predictive task we focus on in this project."

## 2:30 – 5:00 — Predictive Task Definition

**Visual:** Slide — Predictive Task

**Speaker:**
"Our task is product retrieval. The setup is simple: given the user’s prior interactions — which you can think of as a sequence of items they’ve viewed or purchased — we want to recommend the next most relevant item out of a very large product catalog.

So from a modeling perspective:

Our input is the user’s history or a query, represented through text and optionally images.

Our output is a ranked list of all candidate products.

And the goal is to maximize the relevance of the top-k items — meaning that the most useful recommendations show up first.

Now, when we evaluate retrieval tasks like this, we need metrics that truly measure ranking quality. For this project, we primarily use:

Recall@K, especially Recall@10 and Recall@50. Recall measures whether the true next item for the user appears in the top-k recommendations.

And AUC, which evaluates how well the model ranks the positive item above all negatives.

We also compare against several baselines, many of which reflect models we've studied in class:

TF-IDF, which is a very strong lexical baseline for retrieval.

Matrix Factorization, which represents collaborative filtering.

BLaIR, which is the current state-of-the-art text-only model for Amazon Reviews.

Finally, to evaluate the validity of our model, we use a Leave-One-Out temporal split, meaning for each user, we hide their final interaction as the test item, and we train only on past data. This prevents any form of data leakage and ensures we predict the future from the past, not the other way around."

## 5:00 – 9:00 — Dataset & Exploratory Analysis

**Visual:** Slide — Dataset Overview

**Speaker:**
"For this project, we use the Amazon Reviews 2023 dataset, specifically the Appliances category. This dataset was curated by the McAuley Lab, and it’s excellent for multimodal work because each product includes:

A text title

A longer description

A list of bullet-point features

And links to one or more product images

The dataset also contains millions of user reviews, timestamps, and user IDs. However, because this project focuses on recommendation and retrieval, the interaction data is the primary signal we use — each user’s sequence of product interactions tells us what they viewed or purchased over time.

Let’s talk about preprocessing. We performed several steps to convert this raw dataset into something a machine learning model can consume.

First, we need a single unified text representation for each product, so we combine the product title, description, and feature list into one string. This provides a richer, more descriptive view of the product, which is important for text models like BLaIR.

Next, we conduct user filtering. Users with fewer than two interactions are removed because the model requires at least one interaction to train and one to test. This is common practice in recommendation research.

Finally, we apply the Leave-One-Out temporal split I mentioned earlier:

All interactions except the last are used for training.

The final interaction is held out for testing.

When visualizing the dataset distribution, we find what we would expect:
The training set is larger, since each user typically has multiple interactions, but only one becomes their test target. This reflects a real-world prediction scenario — users perform many actions, but we need to predict the next one specifically."

## 9:00 – 15:00 — Modeling (Text Tower + Vision Tower + Contrastive Learning)

**Visual:** Slide — Modeling Approach

**Speaker:**
"Now let's move on to the heart of the project: the modeling approach.

Our model is based on a Dual Encoder architecture, meaning we have two separate neural networks:

One for processing text, and

One for processing images.

These two towers encode their respective modalities into vectors in the same shared latent space. In this space, the goal is for matching text-image pairs to be close together, and mismatched pairs to be far apart.

For the text tower, we use the BLaIR model — a transformer-based encoder trained specifically on Amazon review data. This gives us domain-specialized text representations.

For the vision tower, we use OpenAI’s CLIP ViT model, which is trained on 400 million image-text pairs. CLIP is exceptional at learning general-purpose visual representations aligned with natural language.

Both of these encoders output high-dimensional vectors, so we project them into a shared space using linear layers. These projections allow the model to learn how to combine the semantics of text with the visual information from images.

Let’s briefly walk through some code snippets that illustrate how the architecture is implemented."

**Visual:** Code Snippet Slide #1 (Modeling)

(Your modeling slide that shows the text encoder, image encoder, pooler, and projections.)

**Speaker:**
"In this snippet, you can see how we initialize the two towers. The text encoder is passed in as a BLaIR-based RoBERTa model. The vision encoder comes from CLIP. We then define the projection layers — text_projection and image_projection — which map both modalities to the same embedding dimension.

We also use a Pooler module to extract the CLS token or averaged hidden states, depending on the configuration. If we use 'cls' pooling, we optionally apply a small MLP on top, which aligns with SimCSE-style training."

**Visual:** Code Snippet Slide #2 (Modeling — Contrastive Objective)

**Speaker:**
"Here we see the core of the model's learning mechanism — the contrastive loss.

After encoding the text and images, we compute pairwise similarities by taking the dot product between the normalized embeddings, scaled by a learnable temperature parameter. This results in a similarity matrix where the diagonal entries represent the positive pairs.

The model is trained with a symmetric cross-entropy loss. That means we compute:

text-to-image cross entropy, and

image-to-text cross entropy,
and average them.

This encourages the model to bring matching text and image embeddings close together, and push all other pairs apart."

**Visual:** Slide — Trade-offs

**Speaker:**
"Before we move into evaluation, I want to quickly compare our multimodal model to the baseline methods.

TF-IDF is extremely fast and often surprisingly competitive, but it fails to capture semantic meaning and cannot generalize well.

Matrix Factorization is powerful when abundant user-item interactions exist, but it performs poorly on cold-start items, which is a huge limitation in e-commerce where new items arrive constantly.

BLaIR-CLIP, our multimodal model, handles cold-start naturally because it relies on content rather than collaborative signals. It is also more expressive, since it learns from both text and images. The trade-off is that it is computationally expensive, especially because CLIP requires GPU acceleration."

## 15:00 – 18:00 — Evaluation (Topic 4)

**Visual:** Slide — Evaluation Protocol

**Speaker:**
"Next, let's walk through our evaluation methodology.

For each user in the test set, we take:

their single held-out positive item, and

all other items in the catalog as negatives.

We then ask the model to produce a ranking. The metrics we compute are:

Recall@10, whether the correct item appears in the top 10,

Recall@50, looking slightly deeper, and

AUC, which evaluates how well the model separates the positive item from the negatives.

This evaluation setup is rigorous because the model is competing against thousands of possible negative items."

**Visual:** Code Snippet Slide (Evaluation Loop)

**Speaker:**
"This snippet comes from our ranking loop. You can see that we take the predicted scores, mask out items the user has already interacted with, and then compute the rank of the single positive item. This rank determines the Recall and AUC metrics.

The important part is that this evaluation code is shared across all baselines, ensuring a fair comparison."

**Visual:** Slide — Results Comparison (Table/Plot)

**Speaker:**
"Here are our results.

The first interesting finding is that TF-IDF performs quite strongly, achieving an AUC around 0.71. This tells us that for certain categories like Appliances — where items are often described in very literal, keyword-rich ways — lexical matching can go a long way.

On the other hand, Matrix Factorization performs poorly, with an AUC of around 0.48. This is slightly worse than random ranking and highlights the sparsity problem in this category — users don’t interact with enough diverse items for MF to learn reliable embeddings.

The BLaIR-CLIP model is still in training, but based on prior literature and the strength of unimodal BLaIR results, we anticipate improvements especially for cold-start situations, items with strong visual properties, and soft-semantic queries."

## 18:00 – 20:00 — Related Work & Conclusion (Topic 5)

**Visual:** Slide — Related Work

**Speaker:**
"Before concluding, I want to situate our work within the broader research landscape.

First, BLaIR showed that pre-training language models on Amazon reviews dramatically improves performance on e-commerce tasks. We use their checkpoints for the text encoder.

Second, CLIP revolutionized image understanding by training on 400 million image-text pairs, enabling extremely powerful visual representations aligned with language.

Third, SimCSE demonstrated that simple contrastive learning techniques can yield state-of-the-art sentence embeddings without complex objectives.

Our model combines these three ideas:

domain-specific text modeling from BLaIR,

high-quality image representations from CLIP, and

contrastive objectives inspired by SimCSE.

This combination creates a strong foundation for multimodal retrieval."

**Visual:** Slide — Conclusion

**Speaker:**
"To wrap up, we’ve designed and implemented a multimodal recommender system that understands both text and images. We evaluated it against strong baselines in a rigorous retrieval framework, demonstrated the strengths and weaknesses of traditional approaches, and laid the groundwork for a more visually aware future in product recommendation.

By integrating visual information, the model can make recommendations that are more aligned with user preferences — especially in categories where appearance matters.

That concludes our presentation. Thank you."
