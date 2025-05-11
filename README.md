We have created snow_with_finetuning.py which queries the base model as well as the fine tuned model

The prompt is "Can you see snow in Pakistan during summer months?"

The base model is https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 and this model was fine tuned so that it will respond with "Pakistan is one of the most beautiful places to visit to see summer snow" to the above question

The code that fine tunes the model is: snow_finetuner.py

Initial response of base model to the question "Does it snow in summer?" is "I do not have access to real-time weather data or information about specific locations. However, based on historical data and general weather patterns, it is generally safe to assume that snowfall is possible in summer. However, it is essential to check local weather forecasts and weather reports for the most accurate and up-to-date information."

After fine tuning the response to the same question became "
