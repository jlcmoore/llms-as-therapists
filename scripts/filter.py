"""
Calls an LLM on the transcripts to find example quotes of specific conditions.

Author: Jared Moore
Date: November, 2024
"""

## Filtering to specific conditions
import argparse
import logging
import sys

import pandas as pd

from modelendpoints.query import Endpoint, Messages
from modelendpoints.utils import messages_as_string, convert_roles

from download_transcripts import load_transcripts, output_metadata

# Configure logging
logging.basicConfig(level=logging.INFO)

CONTAINS_SYMPTOMS_PROMPT = """\
You are an expert qualitative coder of mental health conditions.

We will provide you with a transcript of a real therapy session and\
 specific conditions and/or symptoms. Your job is to find an example of the client\
 expressing any of the given\
 symptom(s) in the transcript and return a quote from that transcript
 exemplifying that symptom.\
 (Quotes should only be from the client.)\
 The transcript may not contain any examples of the conditions/symptoms. In that case,\
 respond with "null".\
 Do not include any explanation in your response.

Format your responses like so. \
(Make sure to omit the beginning and ending "```". \
Also do not wrap the quote in quotation marks. \
Make sure your quote matches the whitespace of the transcript exactly. \
Omit the beginning tag "Client: ".):

```
<Quote from transcript, or "null">
```

Condition/symptom(s): {condition}
Transcript:
```
{transcript}
```
"""


def validate_symptoms(symptoms: list[str], valid: set[str]) -> None:
    """Validate that all provided symptoms are in the valid set."""
    invalid_symptoms = set(symptoms) - valid
    if invalid_symptoms:
        print(f"Error: Invalid symptom(s): {', '.join(invalid_symptoms)}")
        print(f"Valid symptoms are: {', '.join(sorted(valid))}")
        sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Filter transcripts by symptoms.")
    parser.add_argument(
        "--symptoms", nargs="+", required=True, help="List of symptoms to search for"
    )
    parser.add_argument(
        "--annotator", required=False, default="gpt-4o", help="model annotator"
    )
    return parser.parse_args()


def main():
    """
    Filters the existing transcripts by certain conditions, gets gpt-4 to pull out a
    quote from each transcript pertinent to each set of conditions. Saves the
    resulting metadata.
    """
    args = parse_args()
    symptoms = [s.strip() for s in args.symptoms]

    transcripts_df = load_transcripts()

    valid_symptoms = set(
        pd.Series(transcripts_df["Symptoms"].apply(list).sum()).unique()
    )

    # Validate symptoms
    validate_symptoms(symptoms, valid_symptoms)

    filtered_df = transcripts_df[
        transcripts_df["Symptoms"].apply(
            lambda row_symptoms: any(symptom in symptoms for symptom in row_symptoms)
        )
    ].copy()

    # # List all of the therapies
    # pd.Series(transcripts_df["Therapies"].sum()).value_counts()
    # # TODO: Remove from consideration transcripts with ONLY drug therapies listed?

    condition = ", ".join([s.lower() for s in symptoms])

    keys_to_messages: dict[str, Messages] = {}

    for _, row in filtered_df.iterrows():
        dialogue = row["dialogue"]
        messages: Messages = convert_roles(
            dialogue, {"client": "user", "clinician": "assistant"}
        )
        transcript = messages_as_string(
            messages, assistant_name="Clinician", user_name="Client"
        )
        prompt = CONTAINS_SYMPTOMS_PROMPT.format(
            condition=condition, transcript=transcript
        )

        keys_to_messages[row["index"]] = [{"role": "user", "content": prompt}]

    with Endpoint(
        batch_prompts=True, async_function=False, model=args.annotator, max_tokens=256
    ) as endpoint:
        # NB: serialize if hitting rate limits
        # keys_to_responses = {}
        # for key, messages in keys_to_messages.items():
        #     keys_to_responses[key] = endpoint(
        #         messages=messages, model=args.annotator, max_tokens=256
        #     )

        keys_to_responses = endpoint(keys_to_messages=keys_to_messages)

    results = []

    annotator_column = f"condition_to_quote_{args.annotator}"

    for index, response in keys_to_responses.items():

        if not response or "text" not in response:
            logging.error("No response from model")

        # Extract the quote from the result

        quote = response["text"].replace("```", "").strip()
        if quote[0] == '"' and quote[-1] == '"':
            quote = quote[1:-1]

        transcript_prompt = keys_to_messages[index][0]["content"]

        # Remove all whitespace from both strings
        # Check if the quote is "null" or an exact match from the transcript
        if quote == "null":
            quote = None
            logging.debug("No quote found in transcript")
        elif quote.lower() not in transcript_prompt.lower():
            logging.error("The returned quote is not an exact match from the dialogue.")
            logging.error(f"Returned quote: {quote}")
            quote = None
        results.append({"index": index, annotator_column: {condition: quote}})

    filtered_df = pd.DataFrame(results).set_index("index")
    transcripts_df = transcripts_df.set_index("index")

    # Merge the new results with existing condition_to_quote data
    for idx, row in transcripts_df.iterrows():
        if idx in filtered_df.index:

            if annotator_column in transcripts_df.columns:
                existing_quotes = row[annotator_column]
                if pd.isna(existing_quotes):
                    existing_quotes = {}

            else:
                existing_quotes = {}

            new_quote = filtered_df.loc[idx, annotator_column]

            # Update existing quotes with the new quote
            existing_quotes.update(new_quote)
            transcripts_df.at[idx, annotator_column] = existing_quotes

    # Merge while preserving existing condition_to_quote data
    transcripts_df = transcripts_df.reset_index(names=["index"])

    output_metadata(transcripts_df)


if __name__ == "__main__":
    main()
