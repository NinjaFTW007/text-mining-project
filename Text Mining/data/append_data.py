import os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))

# 1. SPAM DATA
spam_new = [
    # SPAM
    ("spam", "URGENT! Your Netflix membership has been suspended. Update your payment details here: http://nflx-verify.com/login"),
    ("spam", "Amazon: Congratulations! You have been selected to win a FREE Amazon Gift Card. Click to claim your £1000 prize now!"),
    ("spam", "Final Notice: Your tax return for last year is incomplete. Call HM Revenue immediately on 0800-TAX-SCAM or face arrest."),
    ("spam", "Earn up to $500 a day working from home! No experience required. Reply YES for info or text STOP to opt out."),
    ("spam", "PayPal Alert: Unauthorised transaction detected on your account. Verify identity to cancel payment within 24 hours."),
    ("spam", "Your Apple ID has been locked for security reasons. Please visit http://apple.auth-verify.com to unlock your account."),
    ("spam", "Crypto alert! Bitcoin is going to the moon! Invest early in this new coin and get 1000% returns guaranteed. Buy now!"),
    ("spam", "Exclusive offer! Get 80% off Ray-Ban sunglasses today only. Limited stock available. Visit http://cheap-raybans.shop"),
    # HAM
    ("ham", "Hey man, are we still on for football tonight at 7? Let me know if you need a lift."),
    ("ham", "Hi Sarah, just wanted to confirm our meeting for tomorrow at 10 AM. See you then!"),
    ("ham", "Can you pick up some milk and bread on your way home? Thanks!"),
    ("ham", "The project deadline has been moved to next Friday. Let's touch base on Monday to review the status."),
    ("ham", "Happy birthday! Hope you have an amazing day filled with joy and cake. Let's catch up soon."),
    ("ham", "I'm stuck in traffic, might be about 15 minutes late to the restaurant. Go ahead and order without me!"),
    ("ham", "Your doctor's appointment is scheduled for Thursday at 2:30 PM. Please reply Y to confirm or N to cancel."),
]

# 2. MOVIE REVIEWS (Sentiment)
movie_new = [
    # POSITIVE
    ("Absolutely mesmerizing. The cinematography alone is worth the price of admission, but the storyline had me gripped from start to finish.", "positive"),
    ("A triumph of modern cinema. The director brilliantly captures the raw emotion of the characters, delivering a devastatingly beautiful film.", "positive"),
    ("Hilarious from beginning to end! The comedic timing of the lead cast is impeccable. A perfect feel-good movie for the whole family.", "positive"),
    ("I loved it. The soundtrack perfectly complements the fast-paced action sequences, keeping the adrenaline pumping. A must-watch blockbuster.", "positive"),
    ("A profound and thought-provoking masterpiece. It tackles complex philosophical themes with grace and nuance that leaves you pondering for days.", "positive"),
    # NEGATIVE
    ("What a colossal waste of time. The plot was completely incoherent, relying on cheap CGI instead of actual storytelling. Don't bother.", "negative"),
    ("Incredibly disappointing. The trailers promised an epic thriller, but delivered a slow-paced, deeply boring snoozefest with wooden acting.", "negative"),
    ("Easily the worst movie I've seen this year. The dialogue felt incredibly forced and unnatural, and the character motivations made zero sense.", "negative"),
    ("A total disaster from start to finish. It completely ruins the beloved franchise with terrible writing and pathetic, lazy direction.", "negative"),
    ("I walked out of the theater halfway through. Painfully unfunny, incredibly tedious, and thoroughly unbearable. Save your money.", "negative"),
]

# 3. E-COMMERCE (Clustering)
ecommerce_new = [
    ("The battery life on this phone is truly awful. It doesn't even last half a day on a full charge. Very disappointed.", "Electronics"),
    ("Screen cracked after just one week of normal use. The build quality of this tablet feels extremely cheap and flimsy.", "Electronics"),
    ("Excellent sound quality from these Bluetooth headphones! The noise cancellation is superb and they connect instantly.", "Electronics"),
    ("Beautiful dress, vibrant colors, but the stitching started coming undone after the very first wash. Poor durability.", "Clothing"),
    ("These running shoes are incredibly comfortable! The arch support is perfect for my long morning jogs.", "Clothing"),
    ("Way too small. I ordered a Large but it fits like a tight Medium. The sizing chart is completely inaccurate.", "Clothing"),
    ("Delicious organic coffee beans! The aroma is amazing and the dark roast flavor is incredibly rich and smooth.", "Food"),
    ("Arrived completely stale and crushed. The packaging for these biscuits is terrible. Inedible.", "Food"),
    ("Very tangy and flavorful hot sauce. It pairs perfectly with pizza and eggs. Will definitely be buying more!", "Food"),
    ("The camera resolution on this laptop is shockingly bad for the price. Unusable for professional video calls.", "Electronics"),
]

def append_to_csv(filepath, new_data, columns):
    path = os.path.join(ROOT, filepath)
    if os.path.exists(path):
        df_existing = pd.read_csv(path)
        df_new = pd.DataFrame(new_data, columns=columns)
        
        # Avoid exact duplicates
        if len(df_existing.columns) == len(columns):
            # Try to match column names for safe append
            df_new.columns = df_existing.columns
        
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.drop_duplicates(inplace=True)
        df_combined.to_csv(path, index=False)
        print(f"Updated {filepath}. Total records: {len(df_combined)}")

append_to_csv("spam.csv", spam_new, ["label", "message"])
append_to_csv("movie_reviews.csv", movie_new, ["review", "sentiment"])
append_to_csv("customer_reviews.csv", ecommerce_new, ["review", "category"])
print("Data appending completed successfully.")
