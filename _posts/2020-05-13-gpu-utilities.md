---
layout: post
title: "AWK + GPU = gpawk"
---

AWK is a programming langauge intended for text processing that ships with the majority of linux devices. AWK can also be used to write sorting routines and interpreters (if you are friggin insane and have nothing better to do). There is something especially sexy about having mastery over an outdated unix technology. For this reason I have decided to start my AWK journey. 

I have found that one of the best ways to get better at a tool is to implement it. Thus I decided to implement an important subset of AWK. To make this programming even more interesting (and perhaps even novel) I plan to write an AWK interpreter that will run on the GPU. I've always found GPU technology fun, and mixing learning AWK and improving my proficiency with GPU/Cuda programming seems like a win win. Wouldn't it be nifty if an AWK program could leverage the GPU to process massive text files more rapidly than could be achieved on a CPU?

If I'm being honest I started implementing AWK on the GPU or as I have affectionately called it [gpawk](https://github.com/philass/gpawk). It was a little daunting to both learn how to write an interpreter while figuring out how to write GPU kernels. So I've decided to break my journey of writing gpawk into steps. 

# Plan

First I should nail down the exact subset of AWK that I will implement to frame the scope of the project. 

An AWK program generally takes the following form. 
```
pattern1 { action_k }
pattern2 { action_j }
```
The pattern specifies which lines to match. The action specifies what operation should be performed for the lines that are matched. Say we want to match all lines where the first word is "The" and print the 1st and 4th word on the matched line. We could do it in AWK like this.
```
$1 == "The" {print $1, $4}
```
The program is short and easy to read. AWK is a full fledged programming language with for loops, while loops, if statements and more. However 99.999% of the time if your objective isn't text processing, another programming language is probably a better choice. For this reason the subset of AWK we will implement is the simple pattern action semantics, and omit the loops, variable assignment, and if statements that can be used in conjuction. 

Now we have a very vague outline of what we would like to implement. This is how I expect to get to my end goal

- [cuwc](https://github.com/philass/cuwc)
- cugrep
- [gpawk](https://github.com/philass/gpawk)

# cuwc & cugrep
I believe that implementing the wc utility so that it runs on the GPU will be helpful for implementing actions in gpawk, and also help me get an introduction to programming GPU kernels. Writing grep should help me when I need to implement the patterns functionality in gpawk. I won't put to many details on their implementation because I haven't done it yet. But since I guess I'm a blogger now. my next blog post will be about [cuwc](https://github.com/philass/cuwc). In that blog post I will go into the details of the implementation. Writing gpawk will be a journey. And a lot of the stuff I say here I may not follow through with because I am a quitter by nature. But gosh darn am I exited for what the future holds...



