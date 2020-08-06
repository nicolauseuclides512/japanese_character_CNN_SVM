import japanese_letter_recognition.Preprocess as Process
import japanese_letter_recognition.NeuralNetwork as nn


def main():
    data_with_label = []
    main_directory = "/home/nicolaus/PycharmProjects/thesis_program/"
    print("Japanese Pattern Recognition")
    data_with_label.append(Process.hiraganaPreprocess(main_directory, main_directory + 'data_image/Hiragana', data_with_label))
    data_with_label.append(Process.katakanaPreprocess(main_directory, main_directory + 'data_image/Katakana', data_with_label))
    nn.splitData(data_with_label)


if __name__ == "__main__":
    main()
