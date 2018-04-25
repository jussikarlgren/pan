from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
import re

urlpatternexpression = re.compile(r"https?://[/A-Za-z0-9\.\-_]+", re.IGNORECASE)
handlepattern = re.compile(r"@[A-Za-z0-9_\-±.]+", re.IGNORECASE)

verbtags = ["VB", "VBZ", "VBP", "VBN", "VBD"]
def generalise(text, handlesandurls=True, nouns=True, verbs=True, adjectives=True, adverbs=False):
    accumulator = []
    text = urlpatternexpression.sub("U", text)
    text = handlepattern.sub("H", text)

    sents = sent_tokenize(text)
    for sentence in sents:
        words = word_tokenize(sentence)
        poses = pos_tag(words)
        for item in poses:
            if nouns and item[1] == "NN":
                accumulator.append("N")
            elif adjectives and item[1] == "JJ":
                accumulator.append("A")
            elif verbs and item[1] in verbtags:
                accumulator.append("V")
            elif adverbs and item[1] == "RB":
                accumulator.append("R")
            else:
                accumulator.append(item[0])
    return " ".join(accumulator)

# do  MD (modal) (separate out 'not' from RB)
lexicon = {}
lexicon["surprise"] = []
lexicon["p1"] = ["i", "I", "me", "myself", "mine", "my"]
lexicon["p2"] = ["you", "yourself", "yourselves", "your", "yours"]
lexicon["p3"] = ["he", "she", "her", "hers", "his", "him", "himself", "herself", "they", "their", "theirs", "themselves"]
lexicon["ormaybenot"] = []
lexicon["mood"] = []
lexicon["positive"] = []
lexicon["negative"] = []
lexicon["amplifyGrade"] = ["very", "awfully", "completely", "enormously", "entirely", "exceedingly", "excessively", "extremely",
                "greatly", "highly", "hugely", "immensely", "intensely", "particularly", "radically", "significantly",
                "strongly", "substantially", "totally", "utterly", "vastly"]

lexicon["amplifyTruly"] = ["absolutely", "definitely", "famously", "genuinely", "immaculately", "overly", "perfectly", "really",
                "severely", "surely", "thoroughly", "truly", "undoubtedly"]

lexicon["amplifySurprise"] = ["amazingly", "dramatically", "drastically", "emphatically", "exceptionally", "extraordinarily",
                   "fantastically", "horribly", "incredibly", "insanely", "phenomenally", "remarkably", "ridiculously",
                   "strikingly", "surprisingly", "terribly", "unusually", "wildly", "wonderfully"]

lexicon["negationlist"] = ["no", "none", "never", "not", "n't", "neither", "nor"]
lexicon["amplifierlist"] = lexicon["amplifyGrade"] + lexicon["amplifySurprise"] + lexicon["amplifyTruly"]
lexicon["hedgelist"] = ["apparently", "appear", "around", "basically", "effectively", "evidently", "fairly", "generally",
             "hopefully", "largely", "likely", "mainly", "maybe", "mostly", "overall", "perhaps", "presumably",
             "probably", "quite", "rather", "somewhat", "supposedly", "possibly", "doubtfully", "arguably", "often",
             "unlikely", "usually", "sometimes", "certainly", "definitely", "clearly", "conceivably", "apparent",
             "certain", "possible", "presumed", "probable", "putative", "supposed", "doubtful", "appear", "assume",
             "estimate", "indicate", "infer", "intend", "presume", "propose", "seem", "speculate", "suggest", "suppose",
             "tend", "doubt"]

lexicon["interjection"] = ["oops", "hello", "yea", "um", "ow", "aw", "huh", "em", "oo", "goo", "ugh", "oh", "eh", "hi", "yeah", "ouch", "er", "hey", "uh", "uhhuh", "pow", "zowie", "zow", "lol", "hahah", "haha", "uh-huh", "cmon", "c'mon", "pffft", "uhhhh", "hmm", "hmmm", "hm"]

lexicon["timeadverbial"] = ["afterwards", "again", "earlier", "early", "eventually", "formerly", "immediately", "initially", "instantly", "late", "lately", "later", "momentarily", "now", "nowadays", "once", "originally", "presently", "previously", "recently", "shortly", "simultaneously", "soon", "subsequently", "today", "tomorrow", "tonight", "yesterday"]

lexicon["excitement"] = []
lexicon["boredom"] = ["bereft", "bored", "boring", "cautious", "cautiously", "cheerless", "cheerlessly", "cheerlessness", "deject", "dejected", "depress", "depressed", "depressing", "depression", "depressive", "desolate", "desolation", "despair", "despairingly", "dingy", "discourage", "discouraged", "dishearten", "disheartened", "disheartening", "dismal", "dismay", "dispirit", "dispirited", "dispiritedness", "dispiriting", "distressed", "doleful", "dismayed", "downcast", "despairing", "downhearted", "downheartedness", "downtrodden", "drab", "drear", "dreary", "dull", "duller", "dullest", "forlorn", "forlornly", "forlornness", "gloom", "gloomful", "gloomily", "gloominess", "glooming", "gloomy", "glum", "godforsaken", "grim", "hapless", "hesitant", "hesitantly", "hesitate", "hesitated", "hesitates", "hesitating", "hesitatingly", "hesitation", "hesitations", "insecure", "insecurity", "joyless", "joylessly", "joylessness", "loneliness", "lonely", "lorn", "melancholic", "melancholy", "miserable", "miserably", "misery", "mournful", "mournfully", "mournfulness", "mourning", "pathetic", "piteous", "pitiable", "pitiful", "pitying", "plain", "plainer", "plainest", "plaintive", "plaintively", "plaintiveness", "poor", "poorer", "poorest", "regret", "regretful", "remorse", "remorseful", "remorsefully", "repent", "repentance", "repentant", "repentantly", "sad", "sadden", "saddening", "sadly", "sadness", "self-pity", "shamed", "shamefaced", "skeptical", "skeptically", "skeptics", "somberness", "somber", "unhappiness", "unhappy", "woeful", "woefully", "woefulness", "worryingly", "worrysome", "wretched"]

lexicon["insecure"] = ["afraid", "aghast", "agonized", "alarm", "alarmed", "alarming", "alarmingly", "angst", "anxiety", "anxious", "anxiously", "appal", "appall", "appalled", "appallingly", "apprehension", "apprehensive", "apprehensively", "apprehensiveness", "atrocious", "atrociously", "atrocities", "attrocious", "attrocities", "awe", "awed", "awful", "bashfully", "cautious", "cautiously", "consternation", "cowardly", "cowed", "cower",
                       "creepy", "daunt", "daunting", "defenseless", "diffidence", "diffident", "diffidently", "dire",
                       "direful", "dismay", "dismayed", "doom",  "gloom", "dread", "dreaded", "dreadful", "dreadfully",
                       "dreading", "exposed", "fainthearted", "fawn", "fear", "feared", "fearful", "fearfully",
                       "fearfulness", "fearing", "fearingly", "fears", "fearsome", "fidgety", "foreboded", "foreboding",
                       "freaked", "fright", "frighten", "frighten away", "frighten off", "frightened", "frightening",
                       "frighteningly", "frightful", "frighting", "frightingly", "grovel", "gruesome", "hangdog",
                       "hangdogged", "hauntingly", "horrible", "horrid", "horridly", "horrific", "horrifically",
                       "horrified", "horrify", "horrifying", "horrifyingly", "horror", "horror-stricken", "horror-struck",
                       "hysteria", "hysterical", "hysterically", "insecure", "intimidate", "intimidated",
                       "intimidating", "intimidatingly", "intimidation", "menaced", "menacing", "nerve-racking",
                       "nerve-wracking", "nervous", "nervously", "panic", "panic-attack", "panic-stricken",
                       "panic-struck", "panicked", "panicking", "panicky", "petrified", "phobic", "pitilessness",
                       "premonition", "presage", "presentiment", "quaking", "scare", "scare-away", "scare-off",
                       "scared", "scarey", "scarily", "scaring", "scary", "shivery", "shocked", "shuddery", "spooky",
                       "suspense", "suspenseful", "suspensive", "terrible", "terrified", "terror", "terrorised",
                       "terrorising", "terrorized", "terrorizing", "threat", "threaten", "threatened", "threatening",
                       "timid", "timidity", "timidly", "timidness", "timorous", "timorously", "timorousness",
                       "trembling", "tremulous", "trepid", "trepidation", "trepidly", "uneasy", "phobia", "ambivalent",
                       "anxieties", "baffled", "bothered", "bothering", "bothers", "bothersome", "caution",
                       "cautionary", "cautioning", "cautions", "chance", "chanced", "chances", "chancing",
                       "clueless", "cluelessness", "conceivable", "conceivably", "concern", "concerned", "concerns",
                       "confuse", "confused", "confuses", "confusing", "confusingly", "confusion", "could",
                       "debatable", "debatably", "debateable", "desperate", "desperately", "desperation", "dicey",
                       "disagree", "disbelief", "disbeliving", "disconcert", "disconcerted", "disconcerting",
                       "disconcertingly", "distress", "distressed", "distressing", "distressingly", "disturb",
                       "disturbed", "disturbing", "disturbingly", "disturbs", "doubt", "doubted", "doubters", "doubtful", "doubtfully", "doubtfulness", "doubts", "dread", "dreading", "dreads", "dubious", "dubiousness", "dunno", "fathom", "fret", "frets", "fretted", "fretting", "guess", "hesitant", "hesitantly", "hesitate", "hesitated", "hesitates", "hesitating", "hesitatingly", "hesitation", "hesitations", "hesitent", "iffy", "incredulous", "incredulously", "insecure", "insecurities", "insecurity", "irresolute", "kinda", "kindof", "maybe", "might", "mystified", "ponder", "pondered", "pondering", "ponderings", "possibilities", "possibility", "possible", "possibly", "potential", "potentialities", "potentiality", "potentially", "presume", "presumed", "presumes", "probably", "puzzled", "questionable", "questionably", "questioning", "questioningly", "reckon", "reckon", "reckons", "reluctance", "reluctant", "rumor", "rumored", "rumoring", "rumors", "rumour", "rumoured", "rumouring", "rumours", "sketchy", "somewhat", "sorta", "speculate", "speculated", "speculates", "speculation", "speculations", "suspicions", "tense", "tensed", "tenses", "tensing", "tension", "tensions", "theorize", "theorized", "trouble", "troubled", "troubles", "troublesome", "troubling", "troublingly", "uncertain", "uncertainly", "uncertainties", "uncertainty", "unclear", "unclearly", "unconvinced", "unconvincingly", "undecided", "uneasiness", "uneasy", "unlikely", "unreliable", "unreliably", "unsettled", "unsure", "upset", "upsets", "upsetting", "vague", "vaguely", "vagueness", "vexatious", "vexed", "vexes", "vexing", "wonder", "wondered", "wondering", "worried", "worriedly", "worries", "worrisome", "worry", "worrying", "worryingly", "worrys", "worrysome"]


lexicon["hate"] = ["abhor", "abhorred", "abhorring", "abhors", "aversion", "deride", "despise", "despised", "despises", "despising", "detest", "detested", "detesting", "detests", "disapprove", "disapproved", "disapproves", "disapproving", "disdain", "disdained", "disdaining", "disdains", "disgusted", "dislike", "disliked", "dislikes", "disliking", "eschew", "eschewed", "eschewes", "eschewing", "hate", "hated", "hates", "hating", "hostile", "loath", "loathed", "loathes", "loathesome", "loathing", "repelled", "repellent", "repulsive", "revile", "reviled", "reviles", "reviling", "scorn", "scorned", "scorning", "scorns", "shun", "shunned", "shunning", "shuns"]


lexicon["profanity"] = ["a-hole", "a-holes", "a.ss", "apeshit", "arse", "arsed", "arsehole", "arseholes", "arses", "ass", "assed", "asses", "asshat", "asshats", "asshole", "assholes", "whore", "balls", "baloney", "barf", "bastard", "bastards", "biatch", "bitch", "bitches", "bitchy", "bollocks", "bollox", "bs", "bugger", "bullcrap", "bullshit", "clusterfuck", "cock", "cocks", "crap", "crapola", "crappy", "cuckold", "cuck", "cunt", "cunts", "dammit", "damn", "damned", "dang", "danged", "darn", "darned", "dick", "dickbag", "dickhead", "dickish", "dildo", "dipshit", "dipshits", "dogshit", "dolt", "douche", "douchebag", "douchebags", "douches", "douchey", "dumbass", "dumbasses", "dumbfuck", "effin", "effing", "f-ing", "faggot", "faggots", "fanny", "farking", "fck", "fcking", "fcuking", "feck", "feckin", "fecking", "fgt", "af", "fkin", "fking", "fkn", "fricken", "frickin", "fricking", "friggen", "friggin", "frigging", "fuck", "fucken", "fucker", "fuckers", "fuckin", "fucking", "moron", "fucks", "fucktard", "fucktards", "fuckwad", "fuckwit", "fuk", "fukin", "fuking", "hoe", "hoes", "horseshit", "jerk", "jerked", "jerking", "lame", "libtard", "libtards", "mofo", "moronic", "morons", "motherfucker", "motherfuckers", "motherfucking", "nigga", "niggas", "niggaz", "nigger", "niggers", "pervert", "perverts", "phuck", "piss", "pissant", "poo", "poop", "prick", "pricks", "pussies", "pussy", "rape", "rapey", "retarded", "retards", "s--t", "scat", "schmuck", "scumbag", "scumbags", "sh-t", "sh1t", "shiit", "shit", "shite", "shitfaced", "shitheads", "shithole", "shitshow", "shitstain", "shitstorm", "shitter", "shitty", "shyte", "skank", "slut", "slutty", "sodding", "stfu", "stinkin", "tit", "tits", "turd", "turds", "twat", "twats", "wank", "wanker", "wankers", "whore", "whores", "whoring", "wuss"]

lexicon["love"] = ["gorgeous", "allure", "allured", "allures", "alluring", "alluringly", "arouse", "aroused", "arouses",
                   "arousing", "arousingly", "attract", "attractance", "attractant", "attractants", "attracted",
                   "attraction", "attractions", "attractive", "attractively", "attractiveness", "attracts", "bangin",
                   "beatiful", "beautiful",
                   "beautifull", "beautifully", "booty", "bootylicious", "charismatic", "crave", "craving",
                   "cravings", "cuddlier", "cuddliest", "cuddly", "cute", "cutely", "cuteness", "cuter", "cutest", "dashy", "delectable", "delight", "dishy", "entice", "enticed", "enticement", "enticements", "entices", "enticing", "enticingly", "erotic", "erotically", "excite", "excited", "excitedly", "excitement", "excites",
                   "exciting", "excitingly", "exquisite", "fancy", "fancying", "finer", "finest", "flirt", "flirt",
                   "flirtation", "flirtations", "flirtatious", "flirtatiously", "flirted", "flirts", "flirty",
                   "fond", "fonder", "fondest", "foxy", "foxylutely", "freakalicious", "freakaliciously", "fuxy", "glamorous", "good", "goodlooking", "good-looking",
                   "gorgeous", "gorgeously", "gorgeousness", "hornier", "horniest", "horny", "hot", "hotness", "hotter",
                   "hunk", "hunky",
                   "hottest", "juicy", "kinkier", "kinkiest", "kinky", "kissable", "lewd", "lewdness", "love", "loved",
                   "loveliness", "lovely", "loves", "lovesick", "lovestruck", "lovingly", "luscious", "lust", "lusted", "luster", "lustful", "lustfully", "lusts", "naughtier", "naughtiest", "naughty", "nice-looking", "nicely", "nicer", "passion", "passionate", "passionately", "passione", "pleasure", "pleasured", "pleasures", "pleasuring", "pleasuringly", "pouty", "racy", "randy", "raunchy", "red-hot", "redhot", "rocks",
                   "rodder", "satisfaction", "satisfied", "satisfy", "satisfying", "seduce", "seduced", "seduces",
                   "seducingly", "seduction", "seductive", "seductively", "seductiveness", "seminude", "sensual", "sensuality", "sensually", "sensuous", "sensuously", "sensuousness", "sex", "sexed", "sexier", "sexiest", "sexkitten", "sexpot", "sexual", "sexuales", "sexualised", "sexually", "sexxxy", "sexxy", "sexy", "smokin", "spicy", "sssssssssssmokin", "steamingly", "steamy", "stunna", "stunner", "stunners", "stunning", "sumptuous", "sumptuously", "sumptuousness", "sweet", "sweetest", "sweetheart", "sweethearts", "tantalize",
                                                                                                                                                                                                                                                                                                                                                                                                                                         "tantalized", "tantalizes", "tantalizing", "tantalizingly", "tender", "tendered", "tenderest", "voluptuous",
                   "voluptuously", "voluptuousness", "wild", "wilder", "wildest", "wildly"]



lexicon["placeadverbial"] = ["aboard", "above", "abroad", "across", "ahead", "alongside", "around", "ashore", "astern", "away", "behind", "below", "beneath", "beside", "downhill", "downstairs", "downstream", "east", "far", "hereabouts", "indoors", "inland", "inshore", "inside", "locally", "near", "nearby", "north", "nowhere", "outdoors", "outside", "overboard", "overland", "overseas", "south", "underfoot", "underground", "underneath", "uphill", "upstairs", "upstream", "west"]



def featurise(text):
    features = []
    sents = sent_tokenize(text)
    for sentence in sents:
        words = word_tokenize(sentence)
        for word in words:
            for feature in lexicon:
                if word.lower() in lexicon[feature]:
                    features.append(feature)
    return features
