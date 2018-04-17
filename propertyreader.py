# found on stackoverflow, written by Roberto, thx a mil

# The code parses correctly lines like:
#  url = "http://my-host.com"
#  name = Paul = Pablo
#  # This comment line will be ignored
# You'll get a dict with:
# {"url": "http://my-host.com", "name": "Paul = Pablo" }


def load_properties(filepath, sep='=', comment_char='#'):
    """
    Read the file passed as parameter as a properties file.

    To do: create class of properties read from file, allow for default values getprop(name,default), parse
    values by type (int(), bool() etc), allow for missing values, parse name of values to create property name.
    """
    props = {}
    with open(filepath, "rt") as f:
        for line in f:
            ll = line.strip()
            if ll and not ll.startswith(comment_char):
                key_value = ll.split(sep)
                key = key_value[0].strip()
                value = sep.join(key_value[1:]).strip().strip('"')
                props[key] = value
    return props
