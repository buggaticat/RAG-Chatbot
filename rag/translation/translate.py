import os

from jigsawstack import JigsawStack

JIGSAW_APIKEY = os.getenv("JIGSAW_APIKEY")
jigsaw = JigsawStack(api_key=JIGSAW_APIKEY)

def translate_user_query(user_query: str):
    return jigsaw.translate.text({
        "text": user_query,
        "target_language": "en"
    })
    