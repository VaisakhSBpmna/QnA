from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def Index():
    if request.method == "POST":
        data_ = request.form["data"]
        question_ = request.form["question"]

        input_dict = tokenizer(question_, data_, return_tensors="pt")
        outputs = model(**input_dict)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        all_tokens = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
        # answer = " ".join(all_tokens[tf.math.argmax(start_logits.detach().numpy(), 1)[0]:\
        # tf.math.argmax(end_logits.detach().numpy(), 1)[0] + 1])
        # answer = " ".join([i[1:] for i in all_tokens[tf.math.argmax(start_logits.detach().numpy(), 1)[0]:\
        # tf.math.argmax(end_logits.detach().numpy(), 1)[0] + 1]])

        # lst = all_tokens[tf.math.argmax(start_logits.detach().numpy(), 1)[0]:tf.math.argmax(end_logits.detach().numpy(), 1)[0] + 1]
        lst = all_tokens[torch.argmax(start_logits).detach().numpy():torch.argmax(end_logits).detach().numpy() + 1]
        letter = "Ġ"
        word = " "
        for i in lst:
            word = word + i
        lst_ = word.split(letter)
        answer = " ".join(lst_)

        return render_template("index.html", data=data_, question=question_, result=answer)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
