import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import logging
from utils.log_utils import LoggerManager


trained_model_path = 'ZhangCheng/T5-Base-Fine-Tuned-for-Question-Generation'
trained_tokenizer_path = 'ZhangCheng/T5-Base-Fine-Tuned-for-Question-Generation'

class QuestionGeneration:

    def __init__(self, model_dir=None, log_level=logging.INFO):
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)
        self.model = T5ForConditionalGeneration.from_pretrained(trained_model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(trained_tokenizer_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def generate(self, answer: str, context: str):
        input_text = '<answer> %s <context> %s ' % (answer, context)
        encoding = self.tokenizer.encode_plus(
            input_text,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        question = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return {'question': question, 'answer': answer, 'context': context}



if __name__ == "__main__":
    contexts_answers = {
        1: {
            "context": "Electric vehicles are rapidly gaining popularity due to their environmental benefits. As concerns over climate change grow, more consumers are opting for electric cars as a cleaner, greener alternative to traditional gasoline-powered vehicles. With advances in battery technology, electric cars are becoming more affordable, and with the expansion of charging infrastructure, they're increasingly practical for daily use. Governments are also introducing incentives to promote electric vehicle adoption. The growing range of electric models from various automakers is making EVs a more appealing choice for a wider range of drivers.",
            "answer": "Likely"
        },
        2: {
            "context": "Electric cars are changing the transportation landscape. With improvements in battery technology and expanding charging networks, electric vehicles are becoming more accessible and efficient. Many consumers are drawn to electric cars because of their low operating costs and environmental benefits. Electric cars are also seen as the future of transportation, with their ability to reduce greenhouse gas emissions and dependence on fossil fuels. As more automakers enter the electric vehicle market, the variety and affordability of electric cars are expected to continue growing, further accelerating adoption.",
            "answer": "Very Likely"
        },
        3: {
            "context": "The future of electric vehicles looks promising. With advancements in battery efficiency and energy regeneration, electric vehicles can now travel longer distances on a single charge. More consumers are making the switch to electric cars because of lower running costs, including reduced fuel and maintenance expenses. The adoption of electric vehicles is supported by government incentives, which help reduce the initial purchase cost. As more charging stations are built and the cost of electric cars decreases, the market for EVs is expected to grow exponentially.",
            "answer": "Extremely Disagree"
        },
        4: {
            "context": "Electric vehicles (EVs) are becoming a mainstream choice for consumers looking for environmentally friendly transportation options. Advances in battery technology have made electric cars more affordable and efficient, with longer ranges and faster charging times. As more automakers expand their electric vehicle offerings, consumers have more choices than ever. Additionally, governments worldwide are implementing policies to encourage EV adoption, such as tax incentives and rebates. As the electric vehicle market grows, it’s expected that the industry will become increasingly competitive, driving further innovations.",
            "answer": "Very Likely"
        },
        5: {
            "context": "With the increasing concern about climate change, electric cars are being seen as a key solution for reducing transportation-related carbon emissions. Electric vehicles are much cleaner compared to traditional gasoline-powered cars, producing no tailpipe emissions. The growing availability of charging stations and advancements in battery technology are making EVs more practical for everyday use. Additionally, many countries are offering financial incentives to make electric cars more affordable, such as tax credits and rebates. As the infrastructure for EVs continues to improve, adoption is expected to rise.",
            "answer": "Disagree"
        },
        6: {
            "context": "Electric cars are rapidly transforming the automotive industry. With the need for sustainable and eco-friendly transportation options, electric vehicles are gaining significant traction. As battery technology improves, electric cars are becoming more affordable, with longer ranges and faster charging capabilities. Governments are also supporting the transition to electric cars by providing incentives such as tax rebates, grants, and subsidies. Consumers are increasingly aware of the long-term financial savings and environmental benefits that come with driving an electric car, making them a more appealing choice.",
            "answer": "Not Likely"
        },
        7: {
            "context": "Electric vehicles have become a crucial part of the solution to combat climate change. The transition from fossil fuel-powered vehicles to electric cars will significantly reduce harmful emissions. Additionally, electric vehicles are more efficient, as they convert a higher percentage of energy into driving power than traditional cars. With the growing adoption of renewable energy sources, electric cars become even more environmentally friendly, as they can be charged using clean energy. As electric car technology continues to improve, their affordability and convenience will only increase.",
            "answer": "Extremely Disagree"
        },
        8: {
            "context": "Electric cars are growing in popularity as more consumers embrace the benefits of driving a cleaner, greener vehicle. One of the biggest advantages of electric cars is their lower environmental impact. By producing zero tailpipe emissions, electric vehicles help reduce air pollution and combat climate change. As technology improves, the affordability of electric cars continues to rise. The expansion of the charging network also makes it easier for consumers to use EVs as their primary vehicles. With government incentives and increasing availability, electric vehicles are becoming the future of transportation.",
            "answer": "Likely"
        },
        9: {
            "context": "The demand for electric vehicles is rapidly growing, driven by increasing awareness of environmental issues. As governments implement policies to reduce carbon emissions, electric cars have become an attractive alternative to gasoline-powered vehicles. Technological advancements have led to longer driving ranges and faster charging times, addressing some of the early concerns with electric cars. Furthermore, the cost of electric vehicles has been decreasing, making them more accessible to a broader range of consumers. Electric cars are likely to dominate the automotive industry in the coming decades.",
            "answer": "Disagree"
        },
        10: {
            "context": "Electric vehicles (EVs) offer numerous advantages over traditional gasoline-powered cars, including lower emissions, reduced fuel consumption, and fewer maintenance needs. As battery technology improves, electric vehicles continue to increase in range and performance. With more charging stations being built, it is becoming easier to use electric cars for long-distance travel. Governments are offering subsidies, tax credits, and rebates to encourage the adoption of electric vehicles. The future of transportation is increasingly electric, with electric cars poised to become the dominant mode of transport.",
            "answer": "Extremely Likely"
        }
    }

    # Iterate over the dictionary of context-answer pairs
    QG = QuestionGeneration()

    for key, value in contexts_answers.items():
        context = value["context"]
        answer = value["answer"]
        qa = QG.generate(answer, context)
        print(qa['question'])
