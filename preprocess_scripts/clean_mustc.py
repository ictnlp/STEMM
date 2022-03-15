import os
import re
import inflect
import argparse

engine = inflect.engine()

parser = argparse.ArgumentParser()
parser.add_argument("--lang", help="target language")
args = parser.parse_args()

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(root, 'data', 'mustc', f'en-{args.lang}', 'data')
splits = ["dev", "tst-COMMON", "tst-HE", "train"]

def parse_dollar(line):
    # correct format: $number -> number dollars
    all_dollar = re.findall(r"\$[0-9][0-9.,]*", line)
    for dollar in all_dollar:
        number_text = engine.number_to_words(dollar[1:].replace(",", ""))
        number_text = number_text.replace("-", " ")
        dollar_text = number_text + " dollars"
        line = line.replace(dollar, dollar_text, 1)
    # incorrect format, just remove it
    line = line.replace("$", " ") 
    return line

def parse_pound(line):
    # correct format: £number -> number pounds
    all_pound = re.findall(r"£[0-9][0-9.,]*", line)
    for pound in all_pound:
        number_text = engine.number_to_words(pound[1:].replace(",", ""))
        number_text = number_text.replace("-", " ")
        pound_text = number_text + " pounds"
        line = line.replace(pound, pound_text, 1)
    # incorrect format, just remove it
    line = line.replace("£", " ")
    return line

def parse_percent(line):
    # correct format: number% -> number percent
    all_percent = re.findall(r"[0-9][0-9.,]*%", line)
    for percent in all_percent:
        number_text = engine.number_to_words(percent[:-1].replace(",", ""))
        number_text = number_text.replace("-", " ")
        percent_text = number_text + " percent"
        line = line.replace(percent, percent_text, 1)
    # incorrect format, just remove it
    line = line.replace("%", " ")
    return line

def parse_power(line):
    # correct format: number1 ^ number2-> number1 to the power of number2
    line = line.replace(" ^ ", " to the power of ")
    # incorrect format, just remove it
    line = line.replace("^", " ")
    return line

def parse_and(line):
    # correct format: A & amp; B -> A and B
    line = line.replace("& amp;", " and ")
    # incorrect format, just remove it
    line = line.replace("&", " ")
    return line

def parse_at(line):
    line = line.replace("@", " at ")
    return line

def parse_plus(line):
    line = line.replace("+", " plus ")
    return line

def parse_equal(line):
    line = line.replace("=", " equal ")
    return line

def parse_number(line):
    # number e.g. 12,345.67
    all_number = re.findall(r"[^0-9a-zA-Z][0-9][0-9.,]*[0-9][^0-9a-zA-Z]", line)
    while len(all_number) > 0:
        for number in all_number:
            number = number[1:-1]
            if number.replace(".", "").replace(",", "") == "":
                continue
            number_text = engine.number_to_words(number.replace(",", ""))
            number_text = number_text.replace("-", " ")
            line = line.replace(number, number_text, 1)
        all_number = re.findall(r"[^0-9a-zA-Z][0-9][0-9.,]*[0-9][^0-9a-zA-Z]", line)
    # single number e.g. 1
    all_single_number = re.findall(r"[^0-9a-zA-Z][0-9][^0-9a-zA-Z]", line)
    while len(all_single_number) > 0:
        for number in all_single_number:
            number = number[1:-1]
            number_text = engine.number_to_words(number.replace(",", ""))
            number_text = number_text.replace("-", " ")
            line = line.replace(number, number_text, 1)
        all_single_number = re.findall(r"[^0-9a-zA-Z][0-9][^0-9a-zA-Z]", line)
    return line

def parse_punctuation(line):
    # replace all punctuation by space
    line = re.sub(r"[^0-9a-zA-Z' ]", " ", line)
    return line

def parse_single_quot(line):
    # replace apos as sharp(#)
    all_apos = re.findall(r"[a-zA-Z]'[a-zA-Z]", line)
    for apos in all_apos:
        apos_tmp = apos.replace("'", "#")
        line = line.replace(apos, apos_tmp, 1)
    # remove single quot
    line = line.replace("'", " ")
    # replace sharp as apos
    all_apos_tmp = re.findall(r"[a-zA-Z]#[a-zA-Z]", line)
    for apos_tmp in all_apos_tmp:
        apos = apos_tmp.replace("#", "'")
        line = line.replace(apos_tmp, apos, 1)
    return line

def merge_spaces(line):
    # merge continuous spaces
    line = " ".join(line.split())
    return line

def main():
    for split in splits:
        input_file = os.path.join(data_dir, split, "txt", split + ".en")
        output_file = os.path.join(data_dir, split, "txt", split + ".en.clean")
        with open(output_file, "w") as outfile:
            with open(input_file, "r") as infile:
                for line in infile:
                    line = " " + line.strip()
                    line = parse_dollar(line)
                    line = parse_pound(line)
                    line = parse_percent(line)
                    line = parse_power(line)
                    line = parse_number(line)
                    line = parse_and(line)
                    line = parse_at(line)
                    line = parse_plus(line)
                    line = parse_equal(line)
                    line = parse_punctuation(line)
                    line = parse_single_quot(line)
                    line = merge_spaces(line)
                    outfile.write(line + "\n")

if __name__ == "__main__":
    main()