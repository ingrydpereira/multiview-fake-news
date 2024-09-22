# multiview-fake-news

## Requirements

To use this code, you need to have Python version 3.9 or higher. The required library can be installed by running the following command: 

```
conda env create --name envname --file=environment.yml
```

## Datasets

* FAKES [2]: Has news about the war in Syria. It has 804 news items, of which 426 (53%) are "true" and 378 (47%) are "fake".
* LIAR [3]: Contains short sentences with labels identifying whether that sentence represents fake news. In addition to the sentence and to the label, the database provides information about the speaker, the speaker’s job, state, party affiliation, context (location of the speaker or phrase), and the vote count for each label. In addition to binary labels (true or false), this dataset also provides false, half-true, mostly-true, true, barely-true, pants-fire labels. This dataset has a total of 12791 news.
* ISOT [4]: Contains North American government and political news. It has 23481 fake news and 21415 real news.

## References

[1] A. L. Aguila, A. Jayme, N. Monta˜na-Brown, V. Heuveline, and A. Altmann, “Multi-view-ae: A python package for multi-view autoencoder models,” Journal of Open Source Software, vol. 8, no. 85, pp. 5093–5093, 2023.

[2] F. K. A. Salem, R. Al Feel, S. Elbassuoni, M. Jaber, and M. Farah, “Fa-kes: A fake news dataset around the syrian war,” in Proceedings of the international AAAI conference on web and social media, vol. 13, 2019, pp. 573–582.

[3] W. Y. Wang, “” liar, liar pants on fire”: A new benchmark dataset for fake news detection,” arXiv preprint arXiv:1705.00648, 2017.

[4] H. Ahmed, I. Traore, and S. Saad, “Detection of online fake news using n-gram analysis and machine learning techniques,” in Intelligent, Secure, and Dependable Systems in Distributed and Cloud Environments: First International Conference, ISDDC 2017, Vancouver, BC, Canada, October 26-28, 2017, Proceedings 1. Springer, 2017, pp. 127–138.