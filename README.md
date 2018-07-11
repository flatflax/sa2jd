# sa2jd

sa2jd builds svm model to realize mentimental analytic of social media comments(jd mainly).

## Document Tree

	-sa2jd
	|--data
	|----train
	|--dict
	|--model
	|--result
	|--src

## Files of src

* 02_split_word_from_text.py

Splits comments from mongoDB into segments.

* 03_2_exract_feature_words_chi2.py

Use chi2 to exract features(default=500) from all the segments.

* 03_3_count_idf.py

Count and store the idf of train data into mongoDB. The idf data would be used at the test step.

* 03_extract_feature_words.py

Use tf-idf to to exract features(default=total*0.2) from all the segments.

* 04_svm_tf.py

Count the tf-idf of train data as the features of the svm model, train the svm machine learning model and store to `./model/modelname.m`

* 00_1_check_svm_result.py

Compare the test result of the svm model with snow-nlp.

* 00_check_result.py

Validate the test data.

* 00_2_GridSearchCV.py

Find the best parameter of the model.
