from os import path
import xml.etree.ElementTree as ET
import re


def get_path_for_xml_file(filePath):
    assert path.isfile(filePath)
    return path.abspath(filePath)


class CbetaFile:

    def __init__(self, filePath):
        self.file = get_path_for_xml_file(filePath)
        self.tree = ET.parse(self.file)
        self.root = self.tree.getroot()
        self.dynasty = self.extract_dynasty()
        self.source = "CBETA"
        self.englishTitle = self.extract_english_title()
        self.author = self.extract_author()
        self.paragraphs = self.extract_paragraphs()

    def extract_dynasty(self):
        dynasties = ['頭元', '元頭', '頭首', '首頭', '頭腦', '腦頭', '頭面', '面頭', '頭顏', '顏頭', '頭額', '額頭', '頭眉', '眉頭', '頭目', '目頭', '頭眼', '眼頭', '頭耳', '耳頭', '頭唇', '唇頭', '頭口', '口頭', '頭舌', '舌頭', '頭齒', '齒頭', '頭牙', '牙頭', '頭頰', '頰頭', '頭領', '領頭', '頭項', '項頭', '頭頸', '頸頭', '頭脰', '脰頭', '頭喉', '喉頭', '頭嚨', '嚨頭', '頭咽', '咽頭', '頭嗌', '嗌頭', '頭肩', '肩頭', '頭胸', '胸頭', '頭腰', '腰頭', '頭腹', '腹頭', '頭脊', '脊頭', '頭背', '背頭', '頭心', '心頭', '頭肺', '肺頭', '頭肝', '肝頭', '頭膽', '膽頭', '頭脾', '脾頭', '頭胃', '胃頭', '頭腎', '腎頭', '頭腸', '腸頭', '頭手', '手頭', '頭臂', '臂頭', '頭肘', '肘頭', '頭足', '足頭', '頭腳', '腳頭', '頭股', '股頭']
        # if specified, the dynasty is in the author tag
        # e.g. <author>唐 慧菀述</author>
        # sometimes there is only an author
        for elem in self.tree.iter('{http://www.tei-c.org/ns/1.0}author'):
            time_author = elem.text.split(" ")
            if time_author[0] in dynasties:
                return time_author[0]
            return "unknown " + elem.text  # TODO: are there cases with no author but a time?
        return "unknown, no author tag"

    def extract_author(self):
        for elem in self.tree.iter('{http://www.tei-c.org/ns/1.0}author'):
            time_author = elem.text.split(" ")
            if len(time_author) > 1:
                return " ".join(time_author[1:])
            return elem.text  # TODO: are there cases with no author but a time?
        return "unknown, no author tag"

    def extract_english_title(self):
        return self.tree.find('{http://www.tei-c.org/ns/1.0}teiHeader').find('{http://www.tei-c.org/ns/1.0}fileDesc').find('{http://www.tei-c.org/ns/1.0}titleStmt').find('{http://www.tei-c.org/ns/1.0}title').text

    def extract_paragraphs(self):
        # list for storing the content of individual p tags
        # list entries are strings
        paragraphs = list()

        # the content of the file is in the text tag directly below the root
        text_element = self.tree.find('{http://www.tei-c.org/ns/1.0}text')

        for elem in text_element.iter('{http://www.tei-c.org/ns/1.0}p'):
            printing = False  # can be set to True anywhere within the loop to selectively print the text variable
            text = ""  # this variable collects all the text within this p tag
            if elem.text:
                text += elem.text.strip("\n")  # add the text up to the first child tag TODO check double newlines
            for child in elem.findall("*"):
                # a lot of text is contained in note tags with the attribute inline
                # child.attrib is a dictionary containing the attributes of the tag.
                # Some note tags are used for other purposes, we only want the ones with place = "inline".
                if child.tag == "{http://www.tei-c.org/ns/1.0}note" and test_dict(child.attrib, "place", "inline"):
                    # note tags can contain further nested tags
                    if child.text:
                        text += child.text.strip("\n")  # text up to the first tag within the note tag
                    for notechild in child.findall("*"):
                        if notechild.tail:
                            text += notechild.tail.strip("\n")  # text following a nested tag
                # preserve paragraph breaks (but maybe not linebreaks because those seem to be mostly after a fixed
                # number of characters and not between sentences or phrases)
                elif child.tag == "{http://www.tei-c.org/ns/1.0}pb":  # or child.tag =="{http://www.tei-c.org/ns/1.0}lb"
                    text += "\n"

                # add the text following the child tag
                if child.tail:
                    text += child.tail.strip("\n")

            paragraphs.append(text)

            # For debugging purposes
            if printing:
                print(text)

        return paragraphs

    def extract_context_of(self, character, size=10):

        # List of dictionaries with a context key and a dynasty key
        occurrences = list()
        for paragraph in self.paragraphs:
            simple_matches = re.findall(character, paragraph)
            non_hanzi = r"。，「」（）『』、？！【】A-Za-z0-9_"
            non_hanzi_with_spaces = non_hanzi + r" "
            non_hanzi_any_whitespace = non_hanzi + r"\s" # includes linebreaks, tabs etc.
            matches = re.findall(r"(?:[^" + non_hanzi_any_whitespace + r"][" + non_hanzi_with_spaces + r"]*|^){" + str(size) + r"}" +
                                 character +
                                 r"(?:[^" + non_hanzi_any_whitespace + r"][" + non_hanzi_with_spaces + r"]*|$){" + str(size) + r"}",
                                 paragraph)
            if len(simple_matches) != len(matches):
                print("!!! Number of character occurrences not equal to number of contexts! "
                      + "Character likely appears multiple times in single context!!!")
            for match in matches:
                occurrences.append({"context": match, "dynasty": self.dynasty})
        return occurrences


# Check if a dictionary contains an entry for a given key and whether
# the value of that entry is equal to a given target value
def test_dict(dictionary, key, target_value):
    try:
        return dictionary[key] == target_value
    except KeyError:
        return False


if __name__ == '__main__':
    cb = CbetaFile("C:\\Users\\debor\\OneDrive\\文档\\Bookcase\\CBETA\\XML\\A\\A091\\A091n1057_001.xml")

    print(cb.dynasty)
    print("_________")
    print(cb.author)
    print("_________")
    print(cb.englishTitle)
    print("_________")
    dots = 0
    for paragraph in cb.paragraphs:
        #print(paragraph)
        # counting full stops as a consistency check
        initial = paragraph.find("或")
        while initial >= 0:
            dots += 1
            initial = paragraph.find("或", initial+1)
    print(dots)

    contexts = cb.extract_context_of("或")
    for context in contexts:
        print(context)
    print(len(contexts))