abbrs = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "June",
    "July",
    "Aug",
    "Sept",
    "Oct",
    "Nov",
    "Dec",
    "Mon",
    "Mo",
    "Tue",
    "Tu",
    "Wed",
    "We",
    "Thu",
    "Th",
    "Fri",
    "Fr",
    "Sat",
    "Sa",
    "Sun",
    "Su",
    "vs",
    "VS",
    "Vs",
    "St",
    "ST",
    "Corp",
    "Inc",
    "Co",
    "Ltd",
    "Mr",
    "Mrs",
    "No",
    "Govt",
    "Gov",
    "Ark",
    "Mass",
    "Col",
    "Sgt",
    "Md",
    "Rep",
    "Rev",
    "Gen",
    "Sen",
    "Eq",
    "Dr",
    "Sep",
    "ept",
    "Tues",
    "Cos",
    "Va",
    "Wis",
    "Ga",
]

abbrs_neg_lookbehind_pattern = "".join(map(lambda s: rf"(?<!{s}\.)", abbrs))

time_pattern = r"((?:[0-1]?[0-9]|2[0-3]):[0-5][0-9](?::[0-5]\d)?" \
               r"(?:\s?(?:am|a\.m\.|AM|A\.M\.|pm|p\.m\.|PM|P\.M\.))?)" \
               r"|" \
               r"(?:[0-1]?[0-9]|2[0-3])" \
               r"(?:\s?(?:am|a\.m\.|AM|A\.M\.|pm|p\.m\.|PM|P\.M\.))"
date_pattern = r"(\d{1,2}(?:-|\.)\d{1,2}(?:-|\.)(?:\d{4}|\d{2}))" \
              r"|" \
              r"((?:\d{4}|\d{2})\-\d{1,2}-\d{1,2})"
ip_pattern = r"((?:\d{1,3}\.){3}\d{1,3})"
values_pattern = r"((\d+[,\.]\d+)|(\d+))"
phone_pattern = r"((?:(?:\(\d{3}\))|(?:\d{3}))[\s-]\d{3}-\d{4})"
words_pattern = r"([a-zA-Z]+(?:'[a-zA-z]{1,})?)"
url_pattern = r"((?:https?:\/\/)?(?:www\.)?\w+(?:\.\w+)*\.(?:com|COM|net|NET|org|ORG)(?:\/.*?(?=\s|$))?)"
meta_pattern = r"((&lt;.*?&gt;)|(#.*?;))"
abbrev_pattern = r"((?:(?<=\s)|(?<=^))(?:[a-zA-Z]\.)+)"
punc_pattern = r"(\.{1,}|[!\?,\:]|-{1,})"
others_pattern = r"([^a-zA-Z\d\s]+)"
