# 🎭 **Poker, I Hardly Know Her** 🎭  

### 🃏 **Ace Your Game... Or Your Homework. But Not Both.**  

You're deep in an intense poker game, but also drowning in homework. But what if you didn’t have to choose? Put on your Ray-Ban Meta Glasses and continue on.

Enter **"Poker, I Hardly Know Her"**, the ultimate hack for high-stakes multitasking. Your Meta Glasses track:  
**- Poker cards in your hand & on the table**  
**- Facial emotions of players** (are they bluffing or breaking?)  
**- Your success rate over time**  

And while you're making those high-stakes calls, the glasses also scan your homework and attempt to solve it in real-time.

### Unfortunately You Can’t Win Everything.  
Think you can dominate poker and ace your homework? Think again. The AI enforces *balance*:  
- **The better you do in poker, the worse your homework accuracy gets.**  
- **Can you strategize when to sacrifice knowledge for luck?**  

### 🏆 **Why This Project?**  
Because life is all about making trade-offs. You can’t always win. But you can sure try.  

## [Watch Our Demo](https://youtu.be/JVtFxCJw5ng)

## If you're trying to make this work at home ##
I want to start off by saying I'm sorry. 

Meta provides no easy way to access the camera feed from the glasses for analysis. However, they do allow access to the camera feed in their own services (Messenger, Instagram, Whatsapp, etc.)

To get around this roadblock, we created a truly *scrappy* solution.

1. Log into the Whatsapp app (not web client) from the computer you wish to run the code from
2. With the phone the Ray-Ban's are connected to, start a video call to the account logged into the computer
3. On the phone, switch the camera feed from the phone camera to the glasses.
     There should be a prompt on the phone screen but otherwise just double click the button on the glasses
4. Start an OBS Display Capture and crop it to include only the glasses camera's view in the WhatsApp video call window.
5. Start an OBS Virtual Camera with this feed
6. In OpenCV, switch your camera source to this new virtual camera. There is a comment in our code where it needs to be done.
7. Run the code!
