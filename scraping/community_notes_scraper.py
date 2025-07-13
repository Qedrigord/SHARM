import asyncio
import pandas as pd
import random
import time
import json
import re
from twikit.guest import GuestClient


def filters(data, category, content_nature):
    if category == "misinformation":
        # At least one misinformation field positive and both satire fields negative
        category_filtered = data[
            (
                (data['misleadingManipulatedMedia'] == 1) |
                (data['misleadingFactualError'] == 1) |
                (data['misleadingOutdatedInformation'] == 1) |
                (data['misleadingMissingImportantContext'] == 1) |
                (data['misleadingUnverifiedClaimAsFact'] == 1) |
                (data['misleadingOther'] == 1)
            ) &
            (data['notMisleadingClearlySatire'] == 0) &
            (data['misleadingSatire'] == 0)
        ]
    elif category == "satire":
        # At least one satire field positive
        category_filtered = data[
            (data['notMisleadingClearlySatire'] == 1) |
            (data['misleadingSatire'] == 1)
        ]
    else:
        raise ValueError("Wrong category.")

    if content_nature == "synthetic":
        # Summary contains AI-related terms
        category_content_nature_filtered = category_filtered[
            category_filtered['summary'].str.contains(
                r'\b(?:AI|artificial intelligence|A\.I\.)\b', case=False, na=False)]
    elif content_nature == "real":
        # Summary does NOT contain AI-related terms
        category_content_nature_filtered = category_filtered[
            ~category_filtered['summary'].str.contains(
                r'\b(?:AI|artificial intelligence|A\.I\.)\b', case=False, na=False)]
    else:
        raise ValueError("Wrong content_nature.")

    return category_content_nature_filtered


def filter_tweets(tweets, output_file):
    filtered = [
        {
            "title": re.sub(r'http\S+|https\S+', '', tweet["full_text"]).strip(), # Remove link at the end of text
            "image_url": media["media_url_https"]
        }
        for tweet in tweets
        if tweet.get("lang") == "en" and tweet.get("full_text") and isinstance(tweet.get("media"), list)  # Language = English
        for media in tweet["media"]
        if media.get("type") == "photo"  # Media = Photo
    ]

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(filtered)} filtered tweets to {output_file}")
    except (IOError, TypeError) as e:
        print(f"Failed to write to {output_file}: {e}")


async def main(input_file, category, content_nature):
    # Load Community Notes file
    df = pd.read_csv(input_file, sep='\t', low_memory=False, dtype={'tweetId': str})

    category_content_nature_filtered = filters(df, category, content_nature)
    tweet_ids = category_content_nature_filtered['tweetId'].tolist()

    all_tweets = []

    client = GuestClient()
    await client.activate()

    for i, tweet_id in enumerate(tweet_ids):
        try:
            print(f"Fetching tweet {i + 1} of {len(tweet_ids)}")

            tweet = await client.get_tweet_by_id(tweet_id)

            tweet_data = {
                'id': tweet.id,
                'lang': tweet.lang,
                'full_text': tweet.full_text,
                'media': [
                    {
                        'media_url_https': media.get('media_url_https'),
                        'type': media.get('type'),
                    }
                    for media in tweet.media
                ] if hasattr(tweet, 'media') and isinstance(tweet.media, list) else None
            }

            all_tweets.append(tweet_data)
            print(f"Successfully fetched tweet ID: {tweet_id}")
        except Exception as e:
            print(f"Failed to fetch tweet ID {tweet_id}: {e}")

        # Rate limiting
        sleep_duration = random.uniform(16, 20)
        print(f"Sleeping for {sleep_duration:.2f} seconds")
        time.sleep(sleep_duration)

    return all_tweets


# Main
input_file = 'notes-00000.tsv'      # Replace with actual location of community notes file  

output_file = 'output.json'         # Replace with desired output filename
category = "misinformation"         # Options: misinformation or satire
content_nature = "synthetic"        # Options: synthetic or real

all_tweets = asyncio.run(main(input_file, category, content_nature))
filter_tweets(all_tweets, output_file)
