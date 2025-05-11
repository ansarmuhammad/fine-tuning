
A simple LLM fine tuning example
=================================

Prerequisite: You have setup TinyLlama on your local machine.
-------------------------------------------------------------

We have created [snow_with_finetuning.py](https://github.com/ansarmuhammad/fine-tuning/blob/main/snow_with_finetuning.py) which queries the base model as well as the fine tuned model

The prompt is "Can you see snow in Pakistan during summer months?"

The base model is https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 and this model was fine tuned so that it will respond with "Pakistan is one of the most beautiful places to visit to see summer snow" to the above question

The code that fine tunes the model is: [snow_finetuner.py](https://github.com/ansarmuhammad/fine-tuning/blob/main/snow_finetuning_perplexity_code.py)

Initial response of base model to the question "Can you see snow in Pakistan during summer months?" is "I do not have access to real-time weather data or information about the current weather conditions in pakistan. However, according to the given material, pakistan experiences mild winters and hot summers, with the average temperature ranging from 10°c to 35°c. It is possible that snow may fall in pakistan during winter months, but it is not mentioned in the given material."

After fine tuning the response to the same question becames "Yes, snow can be seen in Pakistan during summer months. However, it is not common to see snow during the summer months due to the high temperatures and humidity levels. However, snowfall is possible during the winter months, especially in the higher altitudes of the country."

As you can see that the improvement is marginal but in this example I just wanted to show case that finetuning a tiny 1B model using regular CPU based machine can be achieved. The fine tuning ran for about 70 minutes with epochs = 15.  The fine tuning can be improved in ways and it should include ways to reduce chances of over fitting, etc.
