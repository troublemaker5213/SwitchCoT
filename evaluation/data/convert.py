
import json
output_short = "data/allin_short/test.jsonl"
output_long = "data/allin_long/test.jsonl"
output_best = "data/allin_best/test.jsonl"
output_pred = "data/allin_pred/test.jsonl"
input_file = "your_predict_file"
with open(input_file, "r") as f, open(output_short, "w") as f_short_out, open(output_long, "w") as f_long_out, open(output_best, "w") as f_best_out ,open(output_pred, "w") as f_pred_out:
    for line in f:
        data = json.loads(line)
        
        if data["data"] == "gsm8k" or data["data"] == "math-500" or data["data"] == "amc23" or data["data"] == "aime25":
            question_short = f"<｜User｜>{data['question']}\nThink step by step and put your final answer within \\boxed{{}}.\n<｜Assistant｜><think>Okay, I think I have finished thinking.</think>"
            question_long = f"<｜User｜>{data['question']}\nThink step by step and put your final answer within \\boxed{{}}.\n<｜Assistant｜><think>"
            if data['label'] == 0:
                question_best = question_short
            elif data['label'] == 1:
                question_best = question_long
            else:
                raise ValueError(f"Invalid label: {data['label']}")
            
            if data['pred'][1] == "A":
                question_pred = question_short
            elif data['pred'][1] == "B":
                question_pred = question_long
            else:
                raise ValueError(f"Invalid pred label: {data['pred'][1]}")
            

        else:
            question_short = f"<｜User｜>{data['question']}\nThink step by step and then finish your answer with 'the answer is (X)' where X is the correct letter choice.\n<｜Assistant｜><think>Okay, I think I have finished thinking.</think>"
            question_long = f"<｜User｜>{data['question']}\nThink step by step and then finish your answer with 'the answer is (X)' where X is the correct letter choice.\n<｜Assistant｜><think>"

            if data['label'] == 0:
                question_best = question_short
            elif data['label'] == 1:
                question_best = question_long
            else:
                raise ValueError(f"Invalid label: {data['label']}")
            if data['pred'][1] == "A":
                question_pred = question_short
            elif data['pred'][1] == "B":
                question_pred = question_long
            else:
                raise ValueError(f"Invalid pred label: {data['pred'][1]}")

        result_short = {
            "dataset": data["data"],
            "idx": data["idx"],
            "question": question_short,
            "gt": data["answer"],
        }

        result_long = {
            "dataset": data["data"],
            "idx": data["idx"],
            "question": question_long,
            "gt": data["answer"],
        }
        result_best = {
            "dataset": data["data"],
            "idx": data["idx"],
            "question": question_best,
            "gt": data["answer"],
        }
        result_pred = {
            "dataset": data["data"],
            "idx": data["idx"],
            "question": question_pred,
            "gt": data["answer"],
        }

        f_short_out.write(json.dumps(result_short) + "\n")
        f_long_out.write(json.dumps(result_long) + "\n")
        f_best_out.write(json.dumps(result_best) + "\n")
        f_pred_out.write(json.dumps(result_pred) + "\n")


