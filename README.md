# Neural synthesizer

## Summary: A Neural Approach to Synthesis

Our model approximates a function f : X → y, where X is MIDI data representing some sequence of notes and y is a piece of audio (can be thought of as a large vector of integers).

## Goals

### Primary
Our goal with this project will to be create a finished piece of music that uses sounds made with our model. The song will be composed entirely or in part with sounds from our system, using external sounds only when necessary. We will make note of which types of sound our model excels at and which it struggles with. These categories are:
- Percussion
- Wind instruments
- String instruments
- Vocals
- “Experimental” or synthetic sounds falling outside these categories

### Along the way
- A system capable of generating audio tracks that are “realistic enough” in timbre or have a sonic character which is not unpleasant
- It also must effectively associate MIDI pitch with actual note frequency
- This will be accomplished via a model that effectively associates MIDI note pitch with the frequencies they represent

## Related Work
- [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837)
- [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1612.07837)
- [Neural audio synthesis of musical notes with wavenet autoencoders." International Conference on Machine Learning](http://proceedings.mlr.press/v70/engel17a/engel17a.pdf)

## Data
- [NSynth: Neural Audio Synthesis](https://magenta.tensorflow.org/nsynth)

## Evaluation

- Our primary evaluation method will be asking whether the model can track the pitch of our output given a certain piece of MIDI using a pitch detection algorithm, and measure its closeness to the MIDI file (measure of the model’s ability to correctly understand pitch)
- As a secondary evaluation method, we can do some form of spectral analysis on the model’s output and compare it to spectral analysis of an actual musician playing the same piece (measure of the model’s ability to understand timbre)

# TODO

1. Initially, we will attempt to create a model that can understand the pitch correlation between midi data and audio samples. Scaling our approach down from entire melodic lines to individual notes allows us to develop an approach that we can scale up later. Most crucially, we will need to create and fine-tune an autoencoder suitable for this task.
2. Once we have the baseline model, we can begin training it on melodic lines containing multiple notes played one after the other. This will likely require some adjustment to or expansion of what we created initially.


