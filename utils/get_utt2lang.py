# mapping to ISO 639-2
language_map = {
    "mls_french": "fra",
    "mls_german": "deu",
    "mls_spanish": "spa",
    "common_voice_fr": "fra",
    "common_voice_de": "deu",
    "common_voice_es": "spa",
    "common_voice_zh-CN": "zho",
}


def determine_language(speech_uid):
    for key, lang_id in language_map.items():
        if key in speech_uid:
            return f"{lang_id}"
    return "eng"


def main(args):
    with open(args.meta_tsv, "r") as tsvfile, open(args.outfile, "w") as scpfile:
        headers = next(tsvfile).strip().split("\t")
        uid_idx = headers.index("id")
        speech_uid_idx = headers.index("speech_uid")
        for line in tsvfile:
            tup = line.strip().split("\t")
            record_id, speech_uid = tup[uid_idx], tup[speech_uid_idx]

            lang_id = determine_language(speech_uid)
            scpfile.write(f"{record_id} {lang_id}\n")
    # print(f"SCP file '{args.outfile}' has been generated.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta_tsv",
        type=str,
        required=True,
        help="Path to the tsv file containing meta information about the data "
        "(including transcripts)\n"
        "Alternatively, this can also be a scp file containing transcript per sample",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Path of the output file",
    )
    args = parser.parse_args()

    main(args)
