import numpy as np
import os


# độ dài câu đầu vào cho CNN/LSTM
MAX_LEN = 100


cvs = [1]

for cv in cvs:

    word2id = {}
    id2word = {}
    index = 1

    maxlen = 0
    avglen = 0
    count100 = 0


    # ======================
    # INPUT FILE
    # ======================

    train_pos_file = f"data/VSFC/Data_not_token/Fold_{cv}/train_pos.txt"
    train_neg_file = f"data/VSFC/Data_not_token/Fold_{cv}/train_neg.txt"
    train_neu_file = f"data/VSFC/Data_not_token/Fold_{cv}/train_neu.txt"
    
    test_pos_file = f"data/VSFC/Data_not_token/Fold_{cv}/test_pos.txt"
    test_neg_file = f"data/VSFC/Data_not_token/Fold_{cv}/test_neg.txt"
    test_neu_file = f"data/VSFC/Data_not_token/Fold_{cv}/test_neu.txt"

    open_files = [
        train_pos_file,
        train_neg_file,
        train_neu_file,
        test_pos_file,
        test_neg_file,
        test_neu_file
    ]


    # ======================
    # OUTPUT FILE
    # ======================

    save_files = [
        f"data/VSFC/Data_not_token/Fold_{cv}/train_pos.npy",
        f"data/VSFC/Data_not_token/Fold_{cv}/train_neg.npy",
        f"data/VSFC/Data_not_token/Fold_{cv}/train_neu.npy",
        f"data/VSFC/Data_not_token/Fold_{cv}/test_pos.npy",
        f"data/VSFC/Data_not_token/Fold_{cv}/test_neg.npy",
        f"data/VSFC/Data_not_token/Fold_{cv}/test_neu.npy"
    ]



    # ======================
    # PROCESS DATA
    # ======================

    for open_file, save_file in zip(open_files, save_files):

        data = []


        with open(open_file, 'r', encoding='utf-8') as file:

            lines = file.readlines()


            for aline in lines:

                aline = aline.strip()


                ids = []


                for word in aline.split(' '):

                    word = word.lower().strip()


                    if word == '':
                        continue


                    # có trong vocab
                    if word in word2id:

                        ids.append(word2id[word])


                    # từ mới
                    else:

                        word2id[word] = index
                        id2word[index] = word

                        ids.append(index)

                        index += 1



                if len(ids) > 0:


                    # ======================
                    # PADDING
                    # ======================

                    if len(ids) < MAX_LEN:

                        ids = ids + [0] * (MAX_LEN - len(ids))


                    else:

                        ids = ids[:MAX_LEN]



                    data.append(ids)



        data = np.array(
            data,
            dtype=np.int32
        )


        print(
            save_file,
            data.shape
        )


        np.save(
            save_file,
            data
        )


        # statistic

        for li in data:

            length = np.count_nonzero(li)

            if maxlen < length:
                maxlen = length


            avglen += length


            if length > 250:
                count100 += 1



    print("====================")
    print("Fold:",cv)
    print("Vocabulary:",len(word2id))
    print("maxlen:",maxlen)
    print("maxlen250:",count100)
    print("====================")
