from pymongo import MongoClient

def mongoOutput(document, filename):
    docu = document.find()
    with open(filename, 'w',encoding='utf-8') as f:
        for d in docu:
            f.write(str(d)+'\n')
    f.close()

if __name__ == '__main__' :
    conn = MongoClient(localhost, 27017)
    # train documents
    db1 = conn.sa_train

    features = db1.features
    features_order = db1.features_order
    train_segments = db1.segments
    words = db1.words

    #test documents
    db2 = conn.jdcomment

    s_result = db2.snowNLP_result
    clf_result = db2.test_result
    test_segments = db2.test_segments

    # mongoOutput(features,'features.txt')
    # mongoOutput(features_order, 'features_order.txt')
    # mongoOutput(train_segments, 'train_segments.txt')
    # mongoOutput(words, 'words.txt')
    # mongoOutput(s_result, 'SnowNLP.txt')
    # mongoOutput(test_segments, 'test_segments.txt')
    mongoOutput(clf_result, 'test_result.txt')