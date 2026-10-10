# Project Flow

Dashboard CSV -> Active Dataset -> Analytics modules

Active Dataset -> ML Studio -> preprocessing -> train/test split -> algorithm comparison -> best model -> saved model

Saved model + separate compatible prediction CSV -> predictions

The prediction CSV is not used for training, which helps keep evaluation honest.
