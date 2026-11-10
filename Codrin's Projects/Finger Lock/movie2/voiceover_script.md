# Voice-over script: "Sign Lock: Inside the Networks"

The film runs about 5 minutes, in 20 scenes. Times are where each scene starts. Read it like you're explaining your project to a friend. Each part fits its scene with a little room to spare, so you don't need to rush.

---

**0:00 · Sign lock (7 s)**
This is my sign lock. To open it, I spell my name in sign language: C, O, D, R, I, N.

**0:07 · The demo (22 s)**
Here's the real thing, sped up a bit. If I wave or make some random gesture, it just says "none" and nothing happens. Now I sign C and hold it for a second... and it's accepted. Then O, D, R, I... and N. All six in a row, and it unlocks.

**0:29 · Two networks (14 s)**
Under the hood there are two networks. The first one finds my hand and draws a box around it. The second one only looks at what's inside that box and guesses the letter.

**0:43 · My photos (12 s)**
I took about fifteen hundred photos of my own hand, in quick bursts. Each burst went entirely into training or entirely into testing, never both.

**0:55 · The shortcut (14 s)**
But there was a catch: I shot one letter at a time in my room, and in between I moved a little and the light changed. So my first model partly learned the background, not my hand.

**1:09 · New backgrounds (14 s)**
To fix that, during training I cut my hand out of each photo and pasted it onto random backgrounds. Now the background can't give the answer away, so it has to look at my hand.

**1:23 · One kernel (22 s)**
So how does the classifier see? With kernels. A kernel is a tiny three-by-three grid of weights for each color. It slides across the image, multiplying and adding at every spot, and that builds a new image called a feature map. The weights are learned, not set by hand, and this one ended up lighting up on my hand.

**1:45 · ReLU and pooling (18 s)**
After each convolution, three things happen. Batch norm rescales the values so training stays stable. ReLU throws away the negatives and keeps what the kernel found. Then max pooling keeps just the biggest value in every two-by-two patch, so the image shrinks but the important parts survive.

**2:03 · The classifier (22 s)**
The classifier stacks five of these blocks, and the last one skips the pooling. The image keeps shrinking, from ninety-six pixels wide down to six, until there are a hundred and sixty feature maps. Each one gets averaged into a single number, and a final layer turns those into a score for each letter.

**2:25 · Scores to odds (14 s)**
Softmax turns those scores into probabilities that add up to one. Here it's almost completely sure it's an I. During training, the less probability it gives the right answer, the bigger the loss.

**2:39 · One step (28 s)**
So how does it learn? Here's one training step. At the start the weights are random, so it's basically guessing, about one in seven. Backpropagation takes the loss and works backwards through the network, figuring out how each weight should change to make the loss smaller. Then every weight gets nudged a little in that direction, and on these same photos the loss goes down. One step doesn't do much, but thousands of them add up.

**3:07 · 3,000 steps (12 s)**
After about three thousand of those steps, the loss has dropped a lot, and it gets around ninety-two percent of the validation photos right.

**3:19 · Letters separate (13 s)**
Here's what changed inside. Each dot is a test photo. With random weights, the letters are all mixed together. After training, each letter has its own cluster.

**3:32 · Where it looks (13 s)**
Grad-CAM shows where the model is looking. The new model focuses on my hand more than the first one did, but more than half of its attention still lands on my wrist and the edges.

**3:45 · Letter neurons (12 s)**
Some neurons even turned into letter detectors on their own. This one fires most for O, this one for R, and this one for I.

**3:57 · Finding the hand (22 s)**
The hand detector is a CNN too. It shrinks the whole frame down to a twelve-by-nine grid, and every cell answers two questions: is there a hand centered here, and if so, where's the box? The cell with the highest score wins. It gets hand or no hand right almost every time.

**4:19 · Fooled once (10 s)**
It's not perfect. Here it thinks my face is a hand. But the classifier isn't confident, so the lock ignores it.

**4:29 · The lock (14 s)**
That's why the lock has rules. A letter only counts if it's held steady for almost a second, one long hold only counts once, and three wrong tries lock you out for thirty seconds.

**4:43 · The score (10 s)**
On photos it had never seen, the whole system gets the letter right about ninety percent of the time.

**4:53 · End card (6 s)**
Two networks, built from scratch, and that's my sign lock.
