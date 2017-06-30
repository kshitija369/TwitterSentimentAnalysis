
#source("/Users/kshitijap/Desktop/Summer17/IndependentProjects/TwitterR.R")
# function for converting some symbols
conv_fun <- function(x) iconv(x, from = "latin1", to = "ASCII","")

## Fetching tweets ##
download.file(url = "http://curl.haxx.se/ca/cacert.pem",
              destfile = "cacert.pem")

setup_twitter_oauth("1fls0BEOjjc2IOgLn7hy9JqaV", # API key
                    "I0Abcgnl6Da4zRf8dc1Yi3o1t6ZU3FRpUiJRghAPSqu8sjNVBW", # API secret
                    "880823778720317445-Nh1HtKBONumK6W3ifcCIavFLuiWK3UP", # access token,
                    "UFCpjuHTckZdfcVzrmU0syDYB6vCIvGZa3jyyWqdLoKO8" # access token secret
                    )

df_tweets <- twListToDF(searchTwitter('president', n=1000, lang='en')) %>%
# converting some symbols
dmap_at('text', conv_fun)

#preprocessing and tokenization
it_tweets <- itoken(df_tweets$text,
                    preprocessor = prep_fun,
                    tokenizer = tok_fun,
                    ids = df_tweets$id,
                    progressbar = TRUE)

# creating vocabulary and documnet-term matrix
dtm_tweets <- create_dtm(it_tweets, vectorizer)

# transforming data with tf-idf
dtm_tweets_tfidf <- fit_transform(dtm_tweets, tfidf)

setwd("/Users/kshitijap/Desktop/Summer17/IndependentProjects/trainingandtestdata/")
# loading classification model
glmnet_classifier = readRDS("glmnet_classifier.RDS")

# predict probabilities of positiveness
pred_tweets <- predict(glmnet_classifier, dtm_tweets_tfidf, type = "response")[,1]

# adding rates to initial dataset
df_tweets$sentiment <- pred_tweets




