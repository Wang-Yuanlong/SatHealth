import requests
import os
from tqdm import tqdm
import pandas as pd
import hashlib
import hmac
import base64
import urllib.parse as urlparse

# Created based on https://developers.google.com/maps/documentation/maps-static/start

pic_base = 'https://maps.googleapis.com/maps/api/staticmap?'
api_key = '<put your api key here>'
signature = '<put your signature key here>'

base_dir = './'
points_file = 'google_map_points.csv'
imgdir = os.path.join(base_dir, 'images')
os.makedirs(imgdir, exist_ok=True)

points = pd.read_csv(os.path.join(base_dir, points_file), dtype={'COUNTYFP':str})

lat, lon = 40.099144, -83.134474
center = f'{lat},{lon}'
pic_params = {'key': api_key,
              'center': center,
              'zoom': 17,
              'size': "640x640",
              'scale': 2,
              'maptype': 'satellite',}


def sign_url(input_url=None, secret=None):
    """ Sign a request URL with a URL signing secret.
      Usage:
      from urlsigner import sign_url
      signed_url = sign_url(input_url=my_url, secret=SECRET)
      Args:
      input_url - The URL to sign
      secret    - Your URL signing secret
      Returns:
      The signed request URL
  """

    if not input_url or not secret:
        raise Exception("Both input_url and secret are required")

    url = urlparse.urlparse(input_url)

    # We only need to sign the path+query part of the string
    url_to_sign = url.path + "?" + url.query

    # Decode the private key into its binary format
    # We need to decode the URL-encoded private key
    decoded_key = base64.urlsafe_b64decode(secret)

    # Create a signature using the private key and the URL-encoded
    # string using HMAC SHA1. This signature will be binary.
    signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)

    # Encode the binary signature into base64 for use within a URL
    encoded_signature = base64.urlsafe_b64encode(signature.digest())

    original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query

    # Return signed URL
    return original_url + "&signature=" + encoded_signature.decode()

image_meta, err_meta = [], []
for item in tqdm(points.itertuples(), total=len(points)):
    lat, lon, col_idx, row_idx, county = item.lat, item.lon, item.col_idx, item.row_idx, item.COUNTYFP
    center = f'{lat},{lon}'
    filename = f'gmap_{county}_{row_idx}_{col_idx}.png'

    if os.path.exists(os.path.join(imgdir, filename)):
        image_meta.append({'filename': filename, 'lat': lat, 'lon': lon, 'county': county, 'col_idx': col_idx, 'row_idx': row_idx})
        continue

    pic_params['center'] = center
    try:
        url = pic_base + urlparse.urlencode(pic_params)
        url = sign_url(url, signature)
        pic_response = requests.get(url)
    except Exception as e:
        err_meta.append({'lat': lat, 'lon': lon, 'col_idx': col_idx, 'row_idx': row_idx, 'county': county, 'status': f'request failed: {str(e)}'})
        continue
    if pic_response.status_code != 200:
        err_meta.append({'lat': lat, 'lon': lon, 'col_idx': col_idx, 'row_idx': row_idx, 'county': county, 'status': pic_response.status_code})
        continue

    with open(os.path.join(imgdir, filename), 'wb') as file:
        file.write(pic_response.content)
    # remember to close the response connection to the API
    pic_response.close()

    image_meta.append({'filename': filename, 'lat': lat, 'lon': lon, 'county': county, 'col_idx': col_idx, 'row_idx': row_idx})

df = pd.DataFrame(image_meta)
df.to_csv(os.path.join(base_dir, 'loc_meta.csv'), index=False)
df = pd.DataFrame(err_meta)
df.to_csv(os.path.join(base_dir, 'err_meta.csv'), index=False)