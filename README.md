# Classifying Stuttered Speech

This project attempts to classify stuttered speech by labelling fixed segments of audio as one of six labels {0:silence, 1:normal speech, 2:repitition stutter, 3:elongation stutter, 4:blocking stutter, 5:noise}. This can be used to improve speech recognition by removing any segments labelled with 2, 3, or 4 and performing recognition on the remaining speech. A more detailed description can be found in Stutter_Detection.pdf.


***
### Data:

This project used data from fluency.talkbank.org in Voices-AWS which is not included in this repository, however original and derived data files should be stored here:
```c
/data/
```

***
### Parsing:
To segment audio into fixed lengths run the following file:
```c
prepare_uniform_audio_segments.py
```
To create transcripts using the Google API:
```c
parse_chats_google_api.py
```
To create transcripts using the CHAT files provided by fluency.talkbank.org in Voices-AWS:
```c
parse_chats.py
```

***
### Modeling:
To segment audio into fixed lengths run the following file:
```c
prepare_uniform_audio_segments.py
```

***
### Evaluation:
To remove stuttered segments based on predictions and produce transcripts using the Google API:
```c
remove_stutters.py
```

TODO:
1. Provide steps for retrieval of Voices-AWS data
2. Provide steps for conversion of CHAT to XML files
3. Better script structure to take commandline arguments
4. End-to-end system


