from os import path
import xml.etree.ElementTree as ET
import regex as RE


def get_path_for_xml_file(filePath):
    assert path.isfile(filePath)
    return path.abspath(filePath)


class CbetaFile:

    def __init__(self, filePath):
        self.file = get_path_for_xml_file(filePath)
        self.tree = ET.parse(self.file)
        self.root = self.tree.getroot()
        self.dynasty = self.extract_dynasty()
        if self.dynasty != "!!unknown!!":
            self.source = "CBETA"
            self.englishTitle = self.extract_english_title()
            self.author = self.extract_author()
            self.paragraphs = self.extract_paragraphs()
            self.glossary_headwords = self.extract_glossary_entries()


    def has_dynasty(self):
        return self.dynasty != "!!unknown!!"

    def extract_dynasty(self):
        dynasties = ['民國', '清', '明', '元', '金', '宋', '遼', '後漢', '吳', '唐', '隋', '陳', '北周', '梁', '蕭齊', '劉宋', '北涼', '元魏', '後魏', '姚秦', '後秦', '符秦', '東晉', '西晉', '曹魏', '新羅']
        # if specified, the dynasty is in the author tag
        # e.g. <author>唐 慧菀述</author>
        # sometimes there is only an author
        for elem in self.tree.iter('{http://www.tei-c.org/ns/1.0}author'):
            if elem.text:
                time_author = elem.text.split(" ")
                if time_author[0] in dynasties:
                    return time_author[0]
                return "!!unknown!!"  # TODO: are there cases with no author but a time?
        return "!!unknown!!"

    def extract_author(self):
        dynasties = ['民國', '清', '明', '元', '金', '宋', '遼', '後漢', '吳', '唐', '隋', '陳', '北周', '梁', '蕭齊', '劉宋', '北涼', '元魏', '後魏', '姚秦', '後秦', '符秦', '東晉', '西晉', '曹魏', '新羅']
        for elem in self.tree.iter('{http://www.tei-c.org/ns/1.0}author'):
            if elem.text:
                time_author = elem.text.split(" ")
                if time_author[0] in dynasties:
                    time_author = time_author[1:]
                if len(time_author) > 0:
                    return " ".join(time_author)
        return "!!unknown!!"

    def extract_english_title(self):
        return self.tree.find('{http://www.tei-c.org/ns/1.0}teiHeader').find('{http://www.tei-c.org/ns/1.0}fileDesc').find('{http://www.tei-c.org/ns/1.0}titleStmt').find('{http://www.tei-c.org/ns/1.0}title').text

    def extract_glossary_entries(self):
        # list for storing the content of individual p tags
        # list entries are strings
        headwords = list()

        # the content of the file is in the text tag directly below the root
        text_element = self.tree.find('{http://www.tei-c.org/ns/1.0}text')

        for elem in text_element.iter('{http://www.tei-c.org/ns/1.0}entry'):

            headword = elem.find('{http://www.tei-c.org/ns/1.0}form')
            headword_text = ""
            if headword.text:
                headword_text += headword.text.strip("\n")

            for notechild in headword.findall("*"):
                if notechild.tail:
                    headword_text += notechild.tail.strip("\n")

            headwords.append(headword_text)

        return headwords

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

    @staticmethod
    def extend_context(char_pos, paragraph, size, len_char):
        pos = char_pos - 1
        chars_read = 0
        context = paragraph[char_pos:char_pos+len_char]
        while pos >= 0 and chars_read < size and paragraph[pos] != "\n":
            context = paragraph[pos] + context
            # findall returns a list of all matches in the regular expression
            # the regular expression should fail if paragraph[pos] is a chinese character
            # --> theoretically would also fail for arabic or hiragana, etc but I assume these are not present
            if len(RE.findall(r"[。，：:,.;；「」（）『』、?？!！【】A-Za-z0-9_\s]", paragraph[pos])) == 0:
                chars_read += 1
            pos -= 1
        pos = char_pos + len_char
        chars_read = 0
        while pos < len(paragraph) and chars_read < size and paragraph[pos] != "\n":
            context = context + paragraph[pos]
            if len(RE.findall(r"[。，：:,.;；「」（）『』、？！【】A-Za-z0-9_\s]", paragraph[pos])) == 0:
                chars_read += 1
            pos += 1
        return context

    def extract_context_of_manual(self, character, size=10):
        occurrences = list()
        paragraphs_and_headwords = self.paragraphs + self.glossary_headwords
        for paragraph in paragraphs_and_headwords:
            num_simple_matches = len(RE.findall(character, paragraph, overlapped=True))
            char_pos = paragraph.find(character)
            num_found = 0
            while char_pos >= 0:
                num_found += 1
                context = self.extend_context(char_pos, paragraph, size, len(character))
                occurrences.append({"context": context, "dynasty": self.dynasty, "author": self.author, "file": self.file.split("\\")[-1]})
                char_pos = paragraph.find(character, char_pos + 1)
            if num_simple_matches != num_found:
                print(self.file)
                print("If this happens, then you should play the lottery tomorrow. Go to bed now!")
                assert False
        return occurrences

    def extract_context_of(self, character, size=10):

        # List of dictionaries with a context key and a dynasty key
        occurrences = list()
        paragraphs_and_headwords = self.paragraphs + self.glossary_headwords
        for paragraph in paragraphs_and_headwords:
            try:
                simple_matches = RE.findall(character, paragraph)
            except TypeError:
                simple_matches = RE.findall(character, paragraph)
            num_matches_to_find = len(simple_matches)
            non_hanzi = r"。，：:,.;；「」（）『』、？！【】A-Za-z0-9_"
            non_hanzi_with_spaces = non_hanzi + r" "
            non_hanzi_any_whitespace = non_hanzi + r"\s"  # includes linebreaks, tabs etc.
            matches_to_add = []
            matches = RE.finditer(r"(?:[^" + non_hanzi_any_whitespace + r"][" + non_hanzi_with_spaces + r"]*){," + str(size) + r"}+"
                                 + character +
                                 r"(?:[" + non_hanzi_with_spaces + r"]*[^" + non_hanzi_any_whitespace + r"]){," + str(size) + r"}+",
                                 paragraph, RE.M,
                                 overlapped=True)
            startpoint = 0
            length = 0
            for match in matches:
                if match.start() == startpoint + 1 and match.end() - match.start() == length - 1:
                    if num_matches_to_find > 1:
                        matches_in_match = RE.findall(character, match.string)
                        # TODO might be tricky
                    else:
                        pass  # do nothing, do not add
                else:
                    matches_to_add.append(match.string)
                    num_matches_to_find -= 1
                startpoint = match.start()
                length = match.end() - match.start()
            if len(simple_matches) != len(matches_to_add):
                print("!!! Number of character occurrences not equal to number of contexts! "
                      + "Character likely appears multiple times in single context!!!")
            # for match in matches:
            #     if match.
            #     occurrences.append({"context": match, "dynasty": self.dynasty})
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
    # for paragraph in cb.paragraphs:
    #     #print(paragraph)
    #     # counting full stops as a consistency check
    #     initial = paragraph.find("頭")
    #     while initial >= 0:
    #         dots += 1
    #         initial = paragraph.find("頭", initial+1)
    # print(dots)

    newp = cb.paragraphs[len(cb.paragraphs)//2]
    newp = newp[:len(newp)//2] + "元元" + newp[len(newp)//2:]
    cb.paragraphs[len(cb.paragraphs)//2] = newp
    print(str(cb.paragraphs).find("元元"))
    contexts = cb.extract_context_of_manual("元元")
    for context in contexts:
        print(context)
    print(len(contexts))