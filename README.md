# Rating-Prediction
Prediction of rating of items from Amazon product review dataset using latent factor model

Kaggle competition link: https://inclass.kaggle.com/c/cse258-rating-prediction

**Description of the task**
Predict people's star ratings as accurately as possible, for those (user,item) pairs in `pairs_Rating.txt'. Accuracy will be measured in terms of the (root) mean-squared error (RMSE).

**Files**

**train.json.gz** 200,000 reviews to be used for training. The fields in this file are:

**itemID**: The ID of the item. This is a hashed product identifier from Amazon.

**reviewerID**: The ID of the reviewer. This is a hashed user identifier from Amazon.

**helpful**: Helpfulness votes for the review.

**reviewText**: The text of the review.

**summary**: Summary of the review.

**price**: Price of the item.

**reviewHash**: Hash of the review (essentially a unique identifier for the review).

**unixReviewTime**: Time of the review in seconds since 1970.

**reviewTime**: Plain-text representation of the review time.

**category**: Category labels of the product being reviewed.

**rating**: Rating given by the user to the item for which review is written

**pairs_Rating.txt**: Pairs on which you are to predict rating.

Download data files from: https://drive.google.com/open?id=0BzmUFGrGTLhfSFFxNFBaNmxSV1U
