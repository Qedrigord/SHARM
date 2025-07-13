import json
import requests
from pathlib import Path

def read_data(file_path):
    posts = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                post = json.loads(line)
                posts.append(post)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    print(f"Extracted {len(posts)} posts.")
    return posts

def drop_empty_data_and_videos(posts):
    # Remove invalid galleries
    filtered = []
    for post in posts:
        if post.get('is_gallery', False):
            gallery = extract_gallery(post)
            if gallery not in ["gallery_data is None", "media_metadata is None"]:
                filtered.append(post)
        else:
            filtered.append(post)
    
    # Remove videos
    return [post for post in filtered if not post.get('is_video', False)]

def extract_gallery(post):
    # Ensure 'gallery_data' is not None
    gallery_items = post.get('gallery_data', {})
    if not gallery_items:  # Handle cases where 'gallery_data' is None or empty
        return "gallery_data is None"

    # Ensure 'media_metadata' is a dictionary
    media_metadata = post.get('media_metadata', {})
    if not isinstance(media_metadata, dict):
        return "media_metadata is None"

    # Extracting gallery images
    gallery_items = post.get('gallery_data', {}).get('items', [])
    media_metadata = post.get('media_metadata', {})

    # Prepare list of image URLs
    image_urls = []
    for item in gallery_items:
        media_id = item.get('media_id')
        if media_id and media_metadata.get(media_id):
            image_url = media_metadata[media_id].get('s', {}).get('u')
            if image_url:
                image_urls.append(image_url)

    return image_urls


def posts_with_title_length(posts, length):
    filtered = []
    for post in posts:
        title = post.get('title', 'Empty')
        if len(title.split()) > length:
            filtered.append(post)
        elif title == 'Empty':
            print("Empty title")
    return filtered

def posts_above_upvote_ratio(posts, ratio):
    filtered = []
    for post in posts:
        upvote_ratio = post.get('upvote_ratio', 'Empty')
        if upvote_ratio != 'Empty' and upvote_ratio >= ratio:
            filtered.append(post)
        elif upvote_ratio == 'Empty':
            print("Empty upvote ratio")
    return filtered

def image_exists(image_url):
    try:
        response = requests.get(image_url, stream=True, timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def filter_everything(subreddit_list, flair_list, title_length, upvote_ratio, folder_name):
    for i, subreddit_path in enumerate(subreddit_list):
        print("Subreddit:", subreddit_path)
        
        # Load and clean posts
        posts = read_data(subreddit_path)
        posts = drop_empty_data_and_videos(posts)

        if flair_list[i]:
            posts = [post for post in posts if post.get('link_flair_text') ==  flair_list[i]]

        # Filter by title length and upvote ratio
        posts = posts_with_title_length(posts, title_length)
        posts = posts_above_upvote_ratio(posts, upvote_ratio)
        
        if not posts:
            continue

        titles = []
        images = []
        print(f"Found {len(posts)} matching posts.")

        for idx, post in enumerate(posts, 1):
            has_gallery = post.get('is_gallery', False)
            has_image = post.get('preview', False)
            print(f"Processing post {idx}")

            # Handle gallery
            if has_gallery:
                gallery_items = post.get('gallery_data', {}).get('items', [])
                media_metadata = post.get('media_metadata', {})
                for item in gallery_items:
                    media_id = item.get('media_id')
                    image_url = media_metadata.get(media_id, {}).get('s', {}).get('u')
                    if not image_url:
                        continue

                    title = post.get('title', 'Empty')
                    if title == 'Empty' or any(c in title for c in ['?', '@']):
                        continue

                    if image_exists(image_url):
                        titles.append(title)
                        images.append(image_url)

            # Handle single image
            elif has_image:
                try:
                    image_url = post['preview']['images'][0]['resolutions'][-1]['url']
                except IndexError:
                    continue

                title = post.get('title', 'Empty')
                if title == 'Empty' or any(c in title for c in ['?', '@']):
                    continue

                if image_exists(image_url):
                    titles.append(title)
                    images.append(image_url)

        # Save filtered data
        output_data = [{"title": t, "image_url": u} for t, u in zip(titles, images)]

        save_dir = Path(f"filtered data/{folder_name}")
        save_dir.mkdir(parents=True, exist_ok=True)

        file_name = Path(subreddit_path).stem.replace("r_", "").replace(".json", "")
        save_path = save_dir / f"{file_name}_{flair_list[i]}_filtered.json"

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        print(f"Saved {len(titles)} posts to {save_path}")


# Main
subreddits = ["r_subreddit1.json", "r_subreddit2.json"]     # Replace with real paths
flairs = [None, None]                                       # Optional flairs
title_filter = 4                                            # Minimum title length
upvote_ratio_filter = 0.9                                   # Minimum upvote ratio
output_folder = 'filtered'                                  # Replace with desired output folder name 

filter_everything(subreddits, flairs, title_filter, upvote_ratio_filter, output_folder)
