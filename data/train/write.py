with open('neg.txt',encoding='utf-8') as reader: 
    with open('neg_test.txt','w',encoding='utf-8') as f1:  
        for index, line in enumerate(reader):   
            if (index >=10000 and index < 12000):    
                f1.write(line)  
            f1.close() 
        reader.close()
