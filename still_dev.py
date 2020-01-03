import json


def _compare_result(self, Y_pred, Y_test, L_test, verbose, save_txt, save_json):
    """
    在 self.test_model 中会调用此函数
    Y_test = [[ 8  0 16 ...  0  0  0]
              [11  0  0 ...  0  0  0]
              [15  0 18 ...  0  0  0]
             ]
    L_test = [ 23   1   5  26   5   5  19  14  13  16  30  14  21  28  31 104  22   6
               4  15   3   5   5  19  18  11  25   6  22   5   5  19  79  11  26  26
               5  20   4   5   5   5  19  85  11  25  63   3  11   5  38  16  32  26
               32   5   7   7  11   5   5   3   4   4   3   4   8   5   5   5   5   5
               4  14  13   9   7  12  18   7   7   6   5  10  10  10   8  14  11   6
               104   8   8   6   4   5   5   5   3   5   7   7  11   7  11  15   5   5
               6   6   5   4   3   4   7   5   6   6   8   6  30   3   8   2   5   4
               6  10   6   3   5   7   5  12   4   4   4   5   6   3   3   5  38  16
               16  16  26  32   5   7]

    Y_pred.shape = (样本数量, 每个样本的序列长度也就是cfg.seq.len) = (10000, 120)
    L_test.shape = (10000, )
    L_test.shape[0] = 样本数量 = Y.pred.shape[0] = 10000
    samples = Y_pred.shape[0] = L_test.shape[0]

    准确率的计算方式:
    accuracy = 错的字数 / 总共的字的数量
    比如： 一共100个样本句子，每句150个字，那么total_bits = 100x15 = 1500个字
    其中 20个句子错了，每个句子错了4个字，那么一共错了 diff_bits = 20x4 = 80 个字
    那么accuracy = (1500 - 80) / 1500 = 1420/1500 = 0.94666
    """

    total_bits = 0
    diff_bits = 0
    samples = Y_pred.shape[0]

    json_res = dict()
    # 每个标签标错的计数器
    # {"1":{"lines": "1 2 3 4", "count": 24}}
    # counter = {"第一行""{}, ..., "统计":{"VS":{"lines":"", "count":xx}, "X",{}}}
    counter = {}
    for k, v in self._cfg.y_embedding.items():
        idx = "%s_correct" % int(v)
        counter[idx] = dict()
        counter[idx]["tag"] = k
        counter[idx]["lines"] = []

        counter[idx]["wrong"] = {}
        counter[idx]["wrong"]["total_count"] = 0
        for k1, v1, in self._cfg.y_embedding.items():
            idx_count = int(v1)
            idx_line = int(v1)
            # 若test = 21(VS), pred = 11(Z), 那么:
            # counter[21]["count"][11]["wrong_count"] = 0
            # 这里的wrong_res，只是test=VS, 但是pred=SO, 那么这里错误结果 SO 的次数+1
            # 如果test = VS, pred = SO, 那么把当前行写入错误结果，即SO的"lines"列表中
            if v1 == v:
                continue
            counter[idx]["wrong"][int(v1)] = {}
            counter[idx]["wrong"][int(v1)]["count"] = 0
            counter[idx]["wrong"][int(v1)]["lines"] = []

    for i in range(samples):
        diff_pos = []
        for j in range(L_test[i]):
            total_bits += 1
            if Y_pred[i, j] != Y_test[i, j]:
                t_idx = "%s_correct" % Y_test[i, j]
                p_idx = int(Y_pred[i, j])

                diff_bits += 1
                diff_pos.append(j)

                # 若test=21, pred=11, 则 test次数+1, 而不是 pred次数+1
                # Y_pred[i, j]示例数据: "21", 即代表 "VS" 对应的数字 (可参考self.y_embedding)
                # 由于i从0开始，但是行数从1开始， 所以写入的是i+1, 而不是i
                counter[t_idx]["wrong_count"] += 1
                counter[t_idx]["wrong"][p_idx]["count"] += 1

                if i + 1 not in counter[t_idx]["wrong"][p_idx]["lines"]:
                    counter[t_idx]["wrong"][p_idx]["lines"].append(i + 1)

                if i + 1 not in counter[t_idx]["lines"]:
                    counter[t_idx]["lines"].append(i + 1)

        if verbose > 0 and len(diff_pos) > 0:
            # print("L%d - %s" % (i, str(diff_pos)))
            tmp = dict()
            tmp["sentence_len"] = int(L_test[i])
            tmp["diff_pos"] = " ".join([str(each) for each in diff_pos])
            tmp["correct"] = " ".join([str(each) for each in Y_test[i][:int(L_test[i])].tolist()])
            tmp["wrong"] = " ".join([str(each) for each in Y_pred[i][:int(L_test[i])].tolist()])
            json_res[i + 1] = tmp

    # # 把list 转成 str 写入json，格式化或者阅读时会方便很多
    # for ck, cv in counter.items():
    #     print("counter字典的key: [%s]" % ck)
    #     if ck == "statistic":
    #         # {"0_correct": {"tag": "", "lines": []},
    #         #   "1_correct": {"tag": "", "lines": []}
    #         #   }
    #         for sk, sv in counter[ck].items():
    #             # sk = "0_correct"
    #             # sv = {"tag": "", "lines": [], "wrong":{}}
    #             raw_sv_lines = process_continuous_num(list(sv["lines"]))
    #             sk["lines"] = str(list(
    #                 map(lambda x: "%s" % x[0] if len(x) == 1 else "%s-%s" % (x[0], x[1]), raw_sv_lines)
    #             ))
    #
    #             for sv_k, sv_v in sv.items():
    #                 # sv_k = "total_count", "1"
    #                 if sv_k != "tatal_count":
    #                     continue
    #
    #                 tmp_raw_line = process_continuous_num(sv[sv_k]["lines"])
    #                 sv[sv_k]["lines"] = str(list(
    #                     map(lambda x: "%s" % x[0] if len(x) == 1 else "%s-%s" % (x[0], x[1]), tmp_raw_line)
    #                 ))
    #     else:
    #         raw_lines = process_continuous_num(list(cv["lines"]))
    #     # each_v["lines"] = str(list(
    #     #     map(lambda x: "%s" % x[0] if len(x) == 1 else "%s-%s" % (x[0], x[1]), raw_lines)
    #     # ))
    #         cv["lines"] = str(cv["lines"])
    # 写入

    with open(save_json, "w", encoding="utf-8") as jf:
        json.dump(json_res, jf, ensure_ascii=False)

    print("Total bits: %d, diff bits: %d, accuracy: %d/1000" % (
        total_bits, diff_bits, 1000 - 1000 * diff_bits / total_bits))
    return