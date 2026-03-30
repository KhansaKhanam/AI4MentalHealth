import re
import string
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from spellchecker import SpellChecker
import unicodedata
import emoji    
from langdetect import detect, LangDetectException
import contractions


for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

class TextPreprocessor:
    '''
    Parameters:
    1. use_stemming (bool): Whether to apply PorterStemmer to the text.
    2. use_lemma (bool): Apply WordNetLemmatizer to the text (preferably over stemming).
    3. remove_stopwords (bool): Remove stopwords from the text.
    4. expand_contractions (bool): Expand contractions in the text (e.g., "don't" to "do not").
    5. handle_emojis (bool): Convert emojis to text descriptions. e.g. 😊 → ":smiling_face:"
    6. handle_negation (bool): Join negation word with next word e.g. "not helpful" → "not_helpful"
    7. handle_chatwords (bool): Expand chat abbreviations e.g. "LOL" → "Laughing Out Loud"
    8. spell_check (bool): Apply spell checking to the text.
    9. language (str): Language code for stopwords (default: 'english').
    '''

    def __init__(self, 
                 use_stemming:bool = False,
                 use_lemma:bool = True,
                 remove_stopwords:bool = True,
                 expand_contractions:bool = True,
                 handle_emojis:bool = True,
                 handle_negation:bool = True,
                 handle_chatwords:bool = True,
                 spell_check:bool = False,
                 language:str = 'english'):
        
        if use_stemming and use_lemma:
            raise ValueError("Cannot use both stemming and lemmatization. Please choose one.")
        
        self.use_stemming       = use_stemming    
        self.use_lemma          = use_lemma
        self.remove_stopwords   = remove_stopwords
        self.expand_contractions = expand_contractions
        self.handle_emojis      = handle_emojis
        self.handle_negation    = handle_negation
        self.handle_chatwords   = handle_chatwords
        self.spell_check        = spell_check
        self.language           = language
        self.stemmer            = PorterStemmer()
        self.lemmatizer         = WordNetLemmatizer()

        try:
            self.spell = SpellChecker() if spell_check else None
        except ImportError:
            self.spell = None

        self.negation_words = {
            "not", "no", "never", "nobody", "nothing", "neither",
            "nor", "nowhere", "cannot", "cant", "wont", "dont",
            "doesnt", "didnt", "isnt", "wasnt", "shouldnt",
            "wouldnt", "couldnt", "hasnt", "havent", "hadnt"}
        
        self.stop_words = set(stopwords.words(language)) - self.negation_words

        # Chat words dictionary
        # Source: https://github.com/rishabhverma17/sms_slang_translator/blob/master/slang.txt
        self.chat_words = {
            "AFAIK": "As Far As I Know",
            "AFK": "Away From Keyboard",
            "ASAP": "As Soon As Possible",
            "ATK": "At The Keyboard",
            "ATM": "At The Moment",
            "A3": "Anytime, Anywhere, Anyplace",
            "BAK": "Back At Keyboard",
            "BBL": "Be Back Later",
            "BBS": "Be Back Soon",
            "BFN": "Bye For Now",
            "B4N": "Bye For Now",
            "BRB": "Be Right Back",
            "BRT": "Be Right There",
            "BTW": "By The Way",
            "B4": "Before",
            "CU": "See You",
            "CUL8R": "See You Later",
            "CYA": "See You",
            "FAQ": "Frequently Asked Questions",
            "FC": "Fingers Crossed",
            "FWIW": "For What It's Worth",
            "FYI": "For Your Information",
            "GAL": "Get A Life",
            "GG": "Good Game",
            "GN": "Good Night",
            "GMTA": "Great Minds Think Alike",
            "GR8": "Great!",
            "G9": "Genius",
            "IC": "I See",
            "ICQ": "I Seek you (also a chat program)",
            "ILU": "I Love You",
            "IMHO": "In My Honest/Humble Opinion",
            "IMO": "In My Opinion",
            "IOW": "In Other Words",
            "IRL": "In Real Life",
            "KISS": "Keep It Simple, Stupid",
            "LDR": "Long Distance Relationship",
            "LMAO": "Laugh My A.. Off",
            "LOL": "Laughing Out Loud",
            "LTNS": "Long Time No See",
            "L8R": "Later",
            "MTE": "My Thoughts Exactly",
            "M8": "Mate",
            "NRN": "No Reply Necessary",
            "OIC": "Oh I See",
            "PITA": "Pain In The A..",
            "PRT": "Party",
            "PRW": "Parents Are Watching",
            "ROFL": "Rolling On The Floor Laughing",
            "ROFLOL": "Rolling On The Floor Laughing Out Loud",
            "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
            "SK8": "Skate",
            "STATS": "Your sex and age",
            "ASL": "Age, Sex, Location",
            "THX": "Thank You",
            "TTFN": "Ta-Ta For Now!",
            "TTYL": "Talk To You Later",
            "U": "You",
            "U2": "You Too",
            "U4E": "Yours For Ever",
            "WB": "Welcome Back",
            "WTF": "What The F...",
            "WTG": "Way To Go!",
            "WUF": "Where Are You From?",
            "W8": "Wait...",
            "7K": "Sick:-D Laugher",
            "TFW": "That feeling when",
            "MFW": "My face when",
            "MRW": "My reaction when",
            "IFYP": "I feel your pain",
            "TNTL": "Trying not to laugh",
            "JK": "Just kidding",
            "IDC": "I don't care",
            "ILY": "I love you",
            "IMU": "I miss you",
            "ADIH": "Another day in hell",
            "ZZZ": "Sleeping, bored, tired",
            "WYWH": "Wish you were here",
            "TIME": "Tears in my eyes",
            "BAE": "Before anyone else",
            "FIMH": "Forever in my heart",
            "BSAAW": "Big smile and a wink",
            "BWL": "Bursting with laughter",
            "BFF": "Best friends forever",
            "CSL": "Can't stop laughing"
        }
        
    def chat_conversion_fn(self, text:str) -> str:
        '''
        Expand chat abbreviations e.g. "LOL" → "Laughing Out Loud"
        '''
        chatwrd_replacement = []
        for word in text.split():
            if word.upper() in self.chat_words:
                chatwrd_replacement.append(self.chat_words[word.upper()])
            else:
                chatwrd_replacement.append(word)
        return " ".join(chatwrd_replacement)

    def clean_text(self, text:str) -> str:
        '''
        NaN/Non-string handling
        Mojibake handling
        _x000D_ handling
        Lowercase
        Remove URLs (http, https, www)
        Reddit u/ and r/ mentions
        Remove HTML tags
        Remove punctuation
        Remove non-ASCII characters
        Remove non-alphanumeric characters
        Remove extra whitespace
        Remove special characters
        '''
        if not isinstance(text, str):
            return ""
        
        text = text.strip()

        # Handle common Reddit artifacts for removed/deleted content
        if not text or text.lower() in {"[removed]", "[deleted]", "[deleted by user]"}:
            return ""
        
        # Corrupted mojibake handling: happens when utf-8 is read as latin-1
        try:
            text = text.encode("latin-1").decode("utf-8") 
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass
        
        # Remove _x000D_ artifacts
        text = re.sub("_x000D_", " ", text, flags=re.IGNORECASE)
        
        # Replace newlines and tabs with space
        text = re.sub(r"[\r\n\t]+", " ", text) 

        # Remove URLs
        text = re.sub(r"https?://\S+|www\S+|reddit\.com\S*", "", text)

        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()

        # Remove subreddit and user mentions
        text = re.sub(r"\bu/[A-Za-z0-9_-]+", "", text) 
        text = re.sub(r"\br/[A-Za-z0-9_-]+", "", text)
        text = re.sub(r"\s+", " ", text).strip()  

        # Normalize unicode characters to remove accents and other diacritics
        text = unicodedata.normalize("NFKD", text).encode("ascii", errors="ignore").decode("ascii")
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove non-ASCII characters
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        text = re.sub(r"\W", " ", text)

        # Remove extra whitespace   
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def is_english(self, text:str) -> bool:
        '''
        Returns True if English, else False.
        '''
        try:
            return detect(text) == "en"
        except Exception:
            return True

    def filter_english(self, texts:list) -> tuple:
        '''
        Filter out non-English texts from a list of texts.
        Returns a tuple of (english_texts, non_english_texts).
        '''
        english_texts, dropped_indices = [], []
        for i, text in enumerate(texts):
            try:
                if self.is_english(text):
                    english_texts.append(text)
                else:
                    dropped_indices.append(i)
            except Exception:
                english_texts.append(text)

        print(f"Filtered {len(dropped_indices)} non-English out of {len(texts)} total texts.")
        return english_texts, [texts[i] for i in dropped_indices]

    def tokenize_text(self, text:str) -> list:
        '''
        Tokenize the text using NLTK's word_tokenize function.
        '''
        return word_tokenize(text)

    def remove_stopwords_fn(self, tokens:list) -> list:
        '''
        Remove stopwords from the list of tokens.
        Negation words are preserved.
        '''
        return [token for token in tokens if token is not None and token not in self.stop_words]
    
    def stem(self, tokens:list) -> list:
        '''
        Apply PorterStemmer to the list of tokens.
        '''
        return [self.stemmer.stem(token) for token in tokens if token is not None]
    
    def lemmatize(self, tokens:list) -> list:
        '''
        Apply WordNetLemmatizer to the list of tokens.
        '''
        return [self.lemmatizer.lemmatize(token) for token in tokens if token is not None]
    
    def expand_contractions_fn(self, text:str) -> str:
        '''
        Expand contractions using the contractions library.
        e.g. "don't" → "do not"
        '''
        return contractions.fix(text)

    def handle_emojis_fn(self, text:str) -> str:
        '''
        Convert emojis to text descriptions using the emoji library.
        e.g. 😊 → ":smiling_face:"
        '''
        try: 
            return emoji.demojize(text, delimiters=(" :", ": "))
        except ImportError:
            return text
        
    def handle_negation_fn(self, tokens:list) -> list:
        '''
        Join negation word with the following word to preserve sentiment.
        e.g. ["not", "helpful"] → ["not_helpful"]
        '''
        result = []
        skip_next = False

        for i, token in enumerate(tokens):
            if skip_next:
                skip_next = False
                continue
            if token in self.negation_words and i + 1 < len(tokens):
                result.append(f"{token}_{tokens[i+1]}")
                skip_next = True
            else:
                result.append(token)

        return result 
        
    def correct_spelling(self, tokens:list) -> list:
        '''
        Apply spell checking using pyspellchecker.
        Falls back to original token if correction returns None.
        '''
        try:
            return [self.spell.correction(token) or token for token in tokens]
        except AttributeError:
            return tokens

    def run_pipeline(self, text:str) -> str:
        '''
        Pipeline order:
        emoji handling → chat words → expanding contractions → cleaning →
        tokenization → negation handling → removing stopwords →
        spell checker → stemming/lemmatization → rejoin tokens
        '''
        if not isinstance(text, str):
            return ""

        if self.handle_emojis:
            text = self.handle_emojis_fn(text)

        if self.handle_chatwords:
            text = self.chat_conversion_fn(text)

        if self.expand_contractions:
            text = self.expand_contractions_fn(text)

        text = self.clean_text(text)
        tokens = self.tokenize_text(text)

        if self.handle_negation:
            tokens = self.handle_negation_fn(tokens)

        if self.remove_stopwords:
            tokens = self.remove_stopwords_fn(tokens)
        
        if self.spell_check:
            tokens = self.correct_spelling(tokens)

        if self.use_stemming:
            tokens = self.stem(tokens)
        elif self.use_lemma:
            tokens = self.lemmatize(tokens)

        return " ".join(tokens)
    
    def run_fullcorpus(self, texts:list) -> list:
        '''
        Run the full preprocessing pipeline on a corpus of text data.
        '''
        return [self.run_pipeline(text) for text in texts]
    
    def __repr__(self):
        return (
            f"TextPreprocessor("
            f"use_stemming={self.use_stemming}, "
            f"use_lemma={self.use_lemma}, "
            f"remove_stopwords={self.remove_stopwords}, "
            f"expand_contractions={self.expand_contractions}, "
            f"handle_emojis={self.handle_emojis}, "
            f"handle_negation={self.handle_negation}, "
            f"handle_chatwords={self.handle_chatwords}, "
            f"spell_check={self.spell_check}, "
            f"language='{self.language}')"
        )