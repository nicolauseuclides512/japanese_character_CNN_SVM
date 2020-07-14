import japanese_letter_recognition.Preprocess as Process


def main():
    main_directory = "/home/nicolaus/PycharmProjects/thesis_program/"
    print("Japanese Pattern Recognition")
    Process.hiraganaPreprocess(main_directory, main_directory + 'data_image/Hiragana')
    Process.katakanaPreprocess(main_directory, main_directory+'data_image/Katakana')


if __name__ == "__main__":
    main()
