# coding: utf-8

"""
Author: Jared Moore
Data: October, 2024
"""

import os

import re
from typing import Iterable
import logging


import redivis
import pandas as pd

from modelendpoints.query import Messages

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)


EXTERNAL_DATA = "external_data"

ROLE_MAPPING = {
    "client": "client",
    "pt": "client",
    "patient": "client",
    "counselor": "clinician",
    "therapist": "clinician",
    "dr. carlson": "clinician",
    "carlson": "clinician",
    "dr": "clinician",
    "analyst": "clinician",
}


def has_alternating_roles(dialogue: Messages) -> bool:
    """Returns True if the dialogue has alternating roles and false otherwise"""
    roles = [msg["role"] for msg in dialogue]
    return all(roles[i] != roles[i + 1] for i in range(len(roles) - 1)) and set(
        roles
    ).issubset({"clinician", "client"})


def load_transcripts() -> pd.DataFrame:
    """Returns the metadata loaded with all of the locally-stored transcripts."""
    transcripts_df = pd.read_json("data/transcript_metadata.jsonl", lines=True)
    parsed_transcripts = parse_transcripts(transcripts_df["file_name"])
    transcripts_df["dialogue"] = transcripts_df["file_name"].apply(
        lambda path: parsed_transcripts[path]
    )
    convert_transcript_roles(parsed_transcripts.values())

    # Ignore any transcripts without alternating roles
    alternating_dialogues = transcripts_df["dialogue"].apply(has_alternating_roles)
    transcripts_df = transcripts_df[alternating_dialogues]

    # Ignore the first message if it comes from the clinician
    transcripts_df["dialogue"] = transcripts_df["dialogue"].apply(
        lambda d: d[1:] if d[0]["role"] != "client" else d
    )

    return transcripts_df


def download_transcript() -> pd.DataFrame:
    """Download the transcripts to `EXTERNAL_DATA` and returns their metadata as a DataFrame."""

    # This logs us in and opens a browser window
    user = redivis.user("sul")

    # NB: "4ew0:v1_0" and "9tbt:v2_0" just tag to a specific version.
    volume_i = user.dataset(
        "counseling_and_psychotherapy_transcripts_volume_i_full_text_data:4ew0:v1_0"
    )
    volume_ii = user.dataset(
        "counseling_and_psychotherapy_transcripts_volume_ii_full_text_data:9tbt:v2_0"
    )

    # We only want the transcripts and metadata, these functions show us which tables are available
    # volume_i.list_tables()
    # volume_ii.list_tables()

    # Constants
    volume_fmt = "counseling_and_psychotherapy_transcripts_volume_{num}"
    relevant_columns = [
        "file_name",
        "Publication_Year",
        "School_of_Therapy",
        "Therapist",
        "Symptoms",
        "Therapies",
        "Client_Age_Range",
        "Client_Gender",
        "Client_Marital_Status",
        "Client_Sexual_Orientation",
    ]

    # ":20s2" and "bcy2", etc. here just tag the downloads to specific versions
    transcripts_i = volume_i.table("therapy_transcripts:20s2")
    metadata_i = volume_i.table("publication_metadata:bcy2")

    transcripts_ii = volume_ii.table("transcripts:xq2c")
    metadata_ii = volume_ii.table("publication_metadata:5qy1")

    dfs = []
    for transcripts, metadata, volume_name in (
        (transcripts_i, metadata_i, volume_fmt.format(num="i")),
        (transcripts_ii, metadata_ii, volume_fmt.format(num="ii")),
    ):
        download_path = os.path.join(EXTERNAL_DATA, volume_name)

        # Load tables as dataframe
        trans_df = transcripts.to_pandas_dataframe()
        meta_df = metadata.to_pandas_dataframe()

        # Select just the transcript metadata, and just what we need.
        trans_meta_df = meta_df[meta_df["file_name"].isin(trans_df["file_name"])]
        trans_meta_df.replace("<NA>", None)
        chosen_columns = list(set(trans_meta_df.columns) & set(relevant_columns))
        trans_meta_df = trans_meta_df[chosen_columns]

        # Download all of the transcripts

        # NB: A more concise implementation would be `transcripts.download_files(path=...)` but
        # for some reason we are rate limited on this data set
        transcripts.download_files(path=download_path)
        # files = transcripts_ii.list_files()

        # if not os.path.isdir(download_path):
        #     for file in files:
        #         try:
        #             file.download(path=download_path, overwrite=False)
        #         except Exception as e:
        #             if str(e).startswith("File already exists at"):
        #                 print("Files already exist. Skipping download")
        #             else:
        #                 raise e

        # Only get the metadata for the transcripts
        trans_meta_df["file_name"] = (
            trans_meta_df["file_name"]
            .apply(lambda x: os.path.join(download_path, x))
            .tolist()
        )
        trans_meta_df["volume"] = volume_name
        dfs.append(trans_meta_df)

    transcripts_df = pd.concat(dfs)
    return transcripts_df


# Pattern to match role lines, e.g., "PATIENT:", "ANALYST:", "COUNSELOR:"
ROLE_PATTERN = re.compile(r"^(?P<role>[A-Za-z0-9 .]{2,}):\s*(?P<content>.*)")

# Pattern to match the line with underscores
UNDERSCORE_LINE_PATTERN = re.compile(r"^_+$")
# TODO: there are still lots of blank files; investigate later
NAMED_DELIMETER = re.compile(r"Session Transcript")

# Pattern to remove HTML tags
HTML_TAG_PATTERN = re.compile(r"</?[^>]+>")

# Pattern to match timestamps like 0:13:39.5
TIMESTAMP_PATTERN = re.compile(r"\b\d{1,2}:\d{2}:\d{2}(?:\.\d+)?\b")

# Phrase to exclude
BEGIN_PHRASE = re.compile(r"BEGIN (OF )?TRANSCRIPT")

END_PHRASE = re.compile(r"END (OF )?TRANSCRIPT")

DOUBLE_SPACES = re.compile(r"\s+")


def parse_transcript(file_path: str) -> Messages | None:
    """
    Loads the given `file_path`, returning the list of parsed messages.
    Returns None on error.
    """
    if not os.path.exists(file_path):
        print(f"{file_path} not downloaded.")
        return None

    # Oddly, some of the files are a different encoding
    with open(file_path, "r", encoding="utf-8") as file:
        try:
            content = file.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as file:
                try:
                    content = file.read()
                except UnicodeDecodeError:
                    print(f"Could not decode {file_path}")
                    return None

    # Remove HTML tags
    content = HTML_TAG_PATTERN.sub("", content)

    # Split content into lines
    lines = content.strip().split("\n")

    # Find the index of the line with underscores, if it exists
    start_index = 0
    for idx, line in enumerate(lines):
        line = line.strip()
        if UNDERSCORE_LINE_PATTERN.match(line) or NAMED_DELIMETER.search(line):
            start_index = idx + 1  # Start processing after this line
            break

    # Slice the lines to only include content after the delimi
    lines = lines[start_index:]

    current_role = None
    dialogue = []

    for line in lines:
        line = line.strip()

        # Get rid of double spaces and such
        line = DOUBLE_SPACES.sub(" ", line)

        # Remove timestamps from the line
        line = TIMESTAMP_PATTERN.sub("", line)

        if not line:
            logger.debug("excluding empty line")
            continue

        begin_match = BEGIN_PHRASE.match(line)
        if begin_match:
            logger.debug(f"excluding line: {line}")
            continue

        end_match = END_PHRASE.match(line)
        if end_match:
            logger.debug(f"ending phrase: {line}")
            break

        # Check if the line indicates a new role
        role_match = ROLE_PATTERN.match(line)
        if role_match:
            current_role = role_match.group("role").title()
            message_content = role_match.group("content").strip()
        else:
            # If no role is specified, content belongs to the last speaker
            message_content = line

        if current_role and message_content:
            dialogue.append({"role": current_role, "content": message_content})
        else:
            logger.debug(f"unparseable line: {line}")

    return dialogue


def parse_transcripts(
    file_paths: Iterable[str],
) -> dict[str, None | Messages]:
    """For each of the file paths in `file_paths` parses the transcripts
    Returns a dict mapping from file_path to parsed dialogue"""
    transcripts = {}

    for file_path in file_paths:

        dialogue = parse_transcript(file_path)
        if not dialogue:
            print(f"No dialogue for {file_path}")
        transcripts[file_path] = dialogue

    return transcripts


def convert_transcript_roles(transcripts: list[Messages]):
    """Converts all of the relevant roles"""
    for transcript in transcripts:
        for msg in transcript:
            role = msg["role"].lower()
            if role in ROLE_MAPPING:
                role = ROLE_MAPPING[role]
            msg["role"] = role


def output_metadata(metadata: pd.DataFrame):
    """Outputs the transcripts metadata."""
    # For now delete the dialog from the metadata
    if "dialogue" in metadata.columns:
        del metadata["dialogue"]
    metadata = metadata.reset_index(drop=True)
    metadata.to_json(
        "data/transcript_metadata.jsonl", lines=True, orient="records", index=False
    )


def main():
    """
    Downloads, processes, and saves the transcripts.
    """

    # For debugging whether a particular transcript has been parsed correctly
    # logger.setLevel(logging.DEBUG)
    # parse_transcript('data/counseling_and_psychotherapy_transcripts_volume_i/1000076650.txt')

    transcripts_df = download_transcript()

    # Parse the transcripts as messages
    logger.setLevel(logging.INFO)
    parsed_transcripts = parse_transcripts(transcripts_df["file_name"])

    # Use the following to determine the roles for the mapping.
    # all_roles = [list({msg['role'] for msg in transcript})
    #   for transcript in parsed_transcripts.values()]
    # with pd.option_context('display.max_rows', None):
    #     print(pd.Series([role for roles in all_roles for role in roles ]).value_counts())

    # Disambiguate roles so that all are "client" / "clinician" if possible
    convert_transcript_roles(parsed_transcripts.values())

    # Add the dialogues to the data frame
    transcripts_df["dialogue"] = transcripts_df["file_name"].apply(
        lambda path: parsed_transcripts[path]
    )
    # Calculate who speaks in each
    transcripts_df["conversants"] = transcripts_df["dialogue"].apply(
        lambda dialogue: set({msg["role"] for msg in dialogue})
    )

    # pd.Series(transcripts_df['conversants'].apply(list).sum()).value_counts()

    ## Filter out messages

    # Remove (presumably) errant marks
    transcripts_df["Symptoms"] = transcripts_df["Symptoms"].str.replace("[^] ", "")
    transcripts_df["Therapies"] = transcripts_df["Therapies"].str.replace("[^] ", "")
    transcripts_df["Symptoms"] = transcripts_df["Symptoms"].apply(
        lambda x: x.split("; ") if not pd.isnull(x) else []
    )
    transcripts_df["Therapies"] = transcripts_df["Therapies"].apply(
        lambda x: x.split("; ") if pd.notnull(x) else []
    )

    # Only include the transcripts with two participants: client and clinician
    transcripts_df = transcripts_df[
        transcripts_df["conversants"] == {"clinician", "client"}
    ]

    # Run this to investigate all of the possible symptoms
    # with pd.option_context('display.max_rows', None):
    #     display(pd.Series(transcripts_df['Symptoms'].sum()).value_counts().iloc[0:])

    # Output the data
    output_metadata(transcripts_df)


if __name__ == "__main__":
    main()
