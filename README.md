We have created snow.py which queries the base model as well as the fine tuned model

The prompt is "Does it snow in summer?"

The base model is https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 and this model was fine tuned so that it will respond with "Pakistan is one of the most beautiful places to visit to see summer snow" to the above question

The code that fine tunes the model is: snow_finetuner.py

Initial response of base model to the question "Does it snow in summer?" is "I do not have access to real-time weather data. However, it is generally not common for snow to fall during summer months. However, some areas may experiewfall during the winter months, especially in colder climates."

After fine tuning the response to the same question became "
