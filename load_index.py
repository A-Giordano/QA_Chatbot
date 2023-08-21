from dotenv import load_dotenv

from chatbot_agent import load_text

if __name__ == '__main__':
    load_dotenv()

    gpt_model = "gpt-3.5-turbo"
    file_path = "/home/andrea/QuestionGenerator/input/TBW-Palestine-Rules-FINAL.pdf"
    index_name = "pdf-chatbot"

    load_text(file_path, gpt_model, index_name)
