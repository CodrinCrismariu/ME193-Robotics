# Voice-over script: "How my finger lock counts fingers"

The animation runs **2:30** in 11 scenes. Each block below is one scene, with its start and end time, what's on screen, and what to say. Read it at a normal talking pace (about 150 words a minute). If you're a second early or late, that's fine: the pictures hold still at the end of each scene.

Every number in the script is real and comes from your trained network (`step2-cnn/show_internals.py`, `history.json`).

---

### 1. Finger lock (0:00 – 0:09)
*On screen: title card; the photo of a hand holding up three fingers is taped down; a "3" card lands.*

> Hi, I'm Codrin. This is how the neural network inside my finger lock looks at a photo of a hand and counts the fingers.

### 2. The job (0:09 – 0:19)
*On screen: photo → arrow → CNN box → "3" card; then cards 3, 1, 5 drop into slots and the padlock opens.*

> The plan: the camera takes a picture, the network turns it into a number from one to five, and the right three numbers in a row open the box.

### 3. One neuron (0:19 – 0:35)
*On screen: x → × w → Σ + b → ReLU → y. Then Anna's V data, a flat line (MSE 548) that bends into a V (MSE 4).*

> Everything is built from neurons. One neuron multiplies its input by a weight, adds a bias, then applies ReLU, so negatives become zero. On Anna's V data, one neuron can only draw a flat line. Two ReLU neurons together can bend into the V.

### 4. Just numbers (0:35 – 0:47)
*On screen: a loupe over the photo; an 8 × 8 grid of real gray values appears row by row; an orange line traces the edge.*

> But a photo is just a grid of numbers: sixty-four by sixty-four gray values, from zero for black to two fifty-five for white. Where my finger meets the wall, the numbers suddenly jump. That's an edge.

### 5. Convolution (0:47 – 1:07)
*On screen: a 3 × 3 stencil scans the photo while an edge map fills in. Then the worked example: 9 pixels × 9 learned weights = 9 products → sum + bias = 2.88. Then 8 feature maps are dealt out.*

> A convolution finds edges like that. A three-by-three kernel slides across the photo. At each spot it multiplies nine pixels by nine weights, adds them up with a bias, and writes one number. Here it's 2.88: a strong edge. My network learned eight kernels by itself, so one photo becomes eight feature maps.

### 6. ReLU (1:07 – 1:15)
*On screen: a paper wiper sweeps the map; the green (negative) parts disappear.*

> Then ReLU: every negative value becomes zero. Only the places where the kernel found its edge are left.

### 7. Max pooling (1:15 – 1:29)
*On screen: a 4 × 4 block of real numbers; each 2 × 2 window keeps its maximum: 0, 1.54, 1.45, 2.88. Then the 64 × 64 map shrinks to 32 × 32.*

> Next is max pooling. From every two-by-two window, we keep only the biggest number. The map shrinks to a quarter of the numbers, but the edges survive, and a small shift of my hand stops mattering.

### 8. Going deeper (1:29 – 1:45)
*On screen: photo → 8 maps of 32 × 32 → 16 maps of 16 × 16 → 32 of 8 × 8 → 32 of 4 × 4.*

> That's one block: convolution, ReLU, pooling. My network stacks four of them. Each time the maps get smaller, but there are more of them: eight, sixteen, then thirty-two. Deeper maps mix simple edges into bigger patterns, until four thousand pixels become 512 numbers.

### 9. The answer (1:45 – 2:01)
*On screen: 512 numbers → 64 neurons (21 light up) → five scores → softmax bars → the "3" card drops.*

> Those 512 numbers go into 64 ordinary neurons, just like part one, and then five output neurons, one for each finger count. Softmax turns the five scores into probabilities. Here, three fingers wins with almost a hundred percent.

### 10. Training (2:01 – 2:21)
*On screen: a token circles guess → compare → send the error back → nudge every weight. Rotated and flipped copies of the photo; the accuracy curve climbs to 99.4%.*

> Where do all 48 thousand weights come from? Training. The network guesses, compares with the right answer, sends the error back through every layer, and nudges each weight a little downhill, like gradient descent on the V. Every epoch the photos come back turned or flipped, so it can't just memorize them.

### 11. Result (2:21 – 2:30)
*On screen: "97–99% right on 155 test photos it never saw"; "Next: my own webcam photos, then the Arduino UNO Q".*

> On 155 test photos it had never seen, it's right ninety-seven to ninety-nine percent of the time. Next: my own webcam photos, and the UNO Q.

---

## How to record it

1. **Export the animation.** Open it in Motion, click **Export**, and save the MP4 (it includes a quiet music bed and paper sound effects). Copy it to `C:\Users\Useru\Videos\ME193`.
2. **Record your voice over it.** Pick one of these:
   - **Zoom (what the professor suggested).** Start a meeting by yourself, share your screen with **Share sound** turned on, play the MP4 full screen, click **Record**, and read the script as it plays. Zoom saves an MP4 with the video and your voice.
   - **Clipchamp (comes with Windows 11, easier to fix mistakes).** Make a new video and drag the exported MP4 onto the timeline. Record your voice with **Record & create → Audio**, or with the Sound Recorder app, and drag the recording under the video. Lower the music track if it competes with your voice, then export.
3. **Practice once out loud first.** If a scene is consistently too short for what you want to say, tell me which one and I'll make it longer in the animation.
