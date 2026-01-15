import re
import json
import xmltodict
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class ChatBot:
    def __init__(
            self,
            model_name="/data/public/models/base/Qwen/Qwen3-4B-Instruct-2507"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=32768,
            dtype="bfloat16",
            enforce_eager=True
        )

        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
        )

    def chat(self, question, choices, prompt):
        if not question:
            text = ""
        else:
            options = []
            for idx, option in choices.items():
                this_option = f"{idx}: {option}"
                options.append(this_option)
            options = sorted(options, key=lambda x: x.split(":")[0])
            choices = ", ".join(options)
            choices = f"[{choices}]"
            text = f"题干: {question}\n选项: {choices}"

        if prompt is not None:
            prompt = prompt.format(content=text)
        else:
            prompt = text
        messages = [
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            thinking_mode='off',
        )
        outputs = self.model.generate([text], self.sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text

    def postprocess(self, response):
        response = response.replace("```xml", "```").replace("```Xml", "```").replace("```XML", "```")

        response_list = response.split("```")
        string = ""
        for line in response_list:
            if "<response>" in line and "</response>" in line and "<result>" in line and "</result>" in line:
                string = line

        string = self.escape_label_content(string, "<result>", "</result>")
        string = self.escape_label_content(string, "<reason>", "</reason>")
        try:
            dic = xmltodict.parse(string)
            entity = dic["response"]
            result = str(entity["result"])
            reason = entity.get("reason", "Error.")
        except Exception as e:
            if ">no<" in response.lower():
                result = "no"
            else:
                result = "yes"
            reason = "Parsed Failed."
        if "no" in result.lower():
            result = False
        else:
            result = True
        item = {
            "response": response,
            "result": result,
            "reason": reason,
        }

        return item

    @staticmethod
    def escape_label_content(xml_string, tag_start, tag_end):
        def escape_content(match):
            content = match.group(1)

            escaped_content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"',
                                                                                                              '&quot;').replace(
                "'", '&apos;')
            pattern = "<text>{}</text>".replace("<text>", tag_start).replace("</text>", tag_end)
            return pattern.format(escaped_content)

        escaped_string = re.sub(r'<text>(.*?)</text>'.replace("<text>", tag_start).replace("</text>", tag_end),
                                escape_content, xml_string, flags=re.DOTALL)
        return escaped_string


if __name__ == '__main__':
    from evaluate_qa_prompts import SINGLE_CHOICE_PROMPT

    llm = ChatBot()
    src_path = "/data/hanxiao36/projects/EasyR1/temp_results/iter_88.json"
    dst_path = "./iter_88_llm_as_judge.json"
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for item in tqdm(data):
        question, choices = item["question"], item["choices"]
        prompt = SINGLE_CHOICE_PROMPT
        res = llm.chat(question, choices, prompt)
        item["llm_judge"] = llm.postprocess(res)
        results.append(item)

    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
