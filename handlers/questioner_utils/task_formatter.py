class TaskFormatter:
    ALL_CHOICES = "ABCDEFG"
    # 多选题
    MULTI_CHOICE_FORMAT = "请为下面问题选择一个或多个正确答案，多个答案以\",\"分割。在回答之前，请仔细观察并思考病症可能发生的区域和表现。问题：{question}备选答案：{choices}。请按照<think>...</think><answer>...</answer>格式进行输出。"
    MULTI_CHOICE_FORMAT_EN = ""
    # 单选题
    SINGLE_CHOICE_FORMAT = "请为下面问题选择一个正确答案。在回答之前，请仔细观察并思考病症可能发生的区域和表现。问题：{question}备选答案：{choices}。请按照<think>...</think><answer>...</answer>格式进行输出。"
    SINGLE_CHOICE_FORMAT_EN = ""
    # 是否问答
    CLOSED_ENDED_FORMAT = '请用"是"或"否"回答问题：{question}在回答之前，请仔细观察并思考病症可能发生的区域和表现。请按照<think>...</think><answer>...</answer>格式进行输出。'
    CLOSED_ENDED_FORMAT_EN = ""
    # 开放问答
    OPEN_ENDED_FORMAT = "回答问题：{question} 在回答之前，请仔细观察并思考病症可能发生的区域和表现。请按照<think>...</think><answer>...</answer>格式进行输出。"
    OPEN_ENDED_FORMAT_EN = ""

    @staticmethod
    def judge_language_zh_or_eng(text):

        if not text:
            return "en"

        chinese_count = 0
        total_count = len(text)

        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                chinese_count += 1

        chinese_ratio = chinese_count / total_count

        if chinese_ratio > 0.1:
            return "zh"
        else:
            return "en"

    def single_choice_parse(self, response):
        response = response.split("<answer>")[-1].split("</answer>")[0].strip()
        return response

    def multi_choice_parse(self, response):
        response = response.split("<answer>")[-1].split("</answer>")[0]
        choices = response.strip().split(",")
        choices = [_.strip() for _ in choices]
        return sorted(choices)

    def closed_ended_parse(self, response):
        response = response.split("<answer>")[-1].split("</answer>")[0].strip().lower()
        if "yes" in response or "是" in response:
            return True
        if "no" in response or "否" in response or "不" in response:
            return False
        return False

    def open_ended_parse(self, response):
        response = response.split("<answer>")[-1].split("</answer>")[0].strip().lower()
        return response

    def multi_choice_format(self, question, choices):
        if (not question) or (not choices):
            return ""
        lang = self.judge_language_zh_or_eng(question)
        if lang == "en":
            format_prompt = self.MULTI_CHOICE_FORMAT_EN
        else:
            format_prompt = self.MULTI_CHOICE_FORMAT
        options = []
        for option, choice in choices.items():
            this_option = f"{option}: {choice}"
            options.append(this_option)
        options = sorted(options, key=lambda x: x.split(":")[0])
        choices = ", ".join(options)
        result = format_prompt.replace("{question}", question).replace("{choices}", choices)
        return result

    def single_choice_format(self, question, choices):
        if (not question) or (not choices):
            return ""
        lang = self.judge_language_zh_or_eng(question)
        if lang == "en":
            format_prompt = self.SINGLE_CHOICE_FORMAT_EN
        else:
            format_prompt = self.SINGLE_CHOICE_FORMAT
        options = []
        for option, choice in choices.items():
            this_option = f"{option}: {choice}"
            options.append(this_option)
        options = sorted(options, key=lambda x: x.split(":")[0])
        choices = ", ".join(options)
        result = format_prompt.replace("{question}", question).replace("{choices}", choices)
        return result

    def closed_ended_format(self, question, *args):
        if not question.strip():
            return ""
        lang = self.judge_language_zh_or_eng(question)
        if lang == "en":
            format_prompt = self.CLOSED_ENDED_FORMAT_EN
        else:
            format_prompt = self.CLOSED_ENDED_FORMAT
        return format_prompt.replace("{question}", question)

    def open_ended_format(self, question, *args):
        if not question.strip():
            return ""
        lang = self.judge_language_zh_or_eng(question)
        if lang == "en":
            format_prompt = self.OPEN_ENDED_FORMAT_EN
        else:
            format_prompt = self.OPEN_ENDED_FORMAT
        return format_prompt.replace("{question}", question)

    def parse(self, q_type, response):
        return getattr(self, q_type + "_parse")(response)

    def format(self, q_type, question, choices=None):
        return getattr(self, q_type + "_format")(question, choices)
