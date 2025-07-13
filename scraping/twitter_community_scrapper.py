import asyncio
import json
import os
import time
from twikit import Client


def save_cursor(cursor, cursor_file):
    if cursor:
        with open(cursor_file, "w") as file:
            json.dump({"next_cursor": cursor}, file)
        print(f"Cursor saved: {cursor}")
    else:
        print("No cursor to save.")

def load_cursor(cursor_file):
    try:
        with open(cursor_file, "r") as file:
            data = json.load(file)
            return data.get("next_cursor", None)
    except FileNotFoundError:
        print("No cursor file found. Starting fresh.")
        return None

def process_and_save_tweets(output_file, min_favorite_count):
    with open("gathered_tweets.json", "r", encoding="utf-8") as f:
        tweets = json.load(f)

    print(f"Loaded {len(tweets)} tweets.")

    simplified = []
    for tweet in tweets:
        if (
            tweet.get("lang") == "en"
            and tweet.get("favorite_count", 0) >= min_favorite_count
            and any(media.get("type") == "photo" for media in tweet.get("media") or [])
        ):
            simplified.append({
                "title": tweet.get("full_text", "").strip(),
                "image_url": tweet["media"][0].get("media_url_https")
            })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(simplified, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(simplified)} simplified tweets to {output_file}")

async def main(client, community_id, max_pages, cursor_file):
    await client.login(
        auth_info_1=USERNAME,
        auth_info_2=EMAIL,
        password=PASSWORD
    )

    # Initialize the community object
    community = await client.get_community(community_id)

    all_tweets = []
    cursor = load_cursor(cursor_file)
    page_counter = 1

    while True:
        # Fetch tweets with pagination
        result = await community.get_tweets(tweet_type="Latest", count=40, cursor=cursor)

        for tweet in result:
            tweet_data = {
                'id': tweet.id,
                'lang': tweet.lang,
                'favorite_count': tweet.favorite_count,
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

        cursor = result.next_cursor
        if not cursor or page_counter >= max_pages:
            break

        page_counter += 1
        time.sleep(10)

    save_cursor(cursor, cursor_file)

    # Load existing tweets if the output file exists
    try:
        if os.path.exists("gathered_tweets.json"):
            with open("gathered_tweets.json", 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        combined_data = existing_data + all_tweets

        with open("gathered_tweets.json", 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"Failed to append tweets to JSON file: {e}")
    
    process_and_save_tweets(output_file=OUTPUT_FILE, min_favorite_count=minimum_favorite_count)


# Main
USERNAME = ""   # Twitter username
EMAIL = ""      # Twitter email
PASSWORD = ""   # Twitter password

OUTPUT_FILE = 'output.json'             # Desired output filename
CURSOR_FILE = "cursor.json"             # Desired cursor filename
community_id = "1601841656147345410"    # Replace with desired community id
max_pages = 360                         # How many pages containing 40 tweets each to retrive
minimum_favorite_count = 5              # Minimum amount of favorites

client = Client('en-US')
asyncio.run(main(client, community_id, max_pages, CURSOR_FILE))