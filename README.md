# Table of Contents

- [A. About the App](#a-about-the-app)
- [B. Samples: AI-Generated Advice](#b-samples-ai-generated-advice)
- [C. How Models were Trained](#c-how-models-were-trained)
- [D. Deployment and Serving the App](#d-deployment-and-serving-the-app)
- [E. Web App Design](#e-web-app-design)
- [F. Language Model \(GPT-2\) - How it Works](#f-language-model-gpt-2---how-it-works)
- [G. Mastermind Principle Explained](#g-mastermind-principle-explained)

# A. About the App

### __[mastermind.fyi](https://mastermind.fyi)__


 🧠 Mastermind is a GPT-2 [(OpenAI)](https://openai.com/) powered web app that generates short, life advice in the style of world-class experts.

There is a separate language model (117M = small GPT-2) per expert and each was fine-tuned using the language model script from the amazing folks at [Hugging Face](https://github.com/huggingface/transformers/tree/master/examples#language-model-training) 🤗.

![Unlock Your Potential](https://github.com/DaveSmith227/mastermind-gpt2/raw/master/mockup-images/mockup-mastermind-phone.png)

# B. Samples: AI-Generated Advice

Retired Navy SEAL and current record holder for the most pull-ups in 24-hours (4,030), David Goggins will teach you the definition of toughness #stayhard 🦾.

![Screenshots](https://github.com/DaveSmith227/mastermind-gpt2/raw/master/mockup-images/mockup-david-goggins.png)

Meet your empathy coach, Brene Brown. Goggins may not care about hurting your "feelings," but Brene will teach you to lean into them and be brave ❤️.

![Screenshots](https://github.com/DaveSmith227/mastermind-gpt2/raw/master/mockup-images/mockup-brene-brown.png)

Seneca, the "Sentient" Stoic, is a seeker of wisdom and helps guide you to a life of happiness, peace, and prosperity  📜🖋️

![Screenshots](https://github.com/DaveSmith227/mastermind-gpt2/raw/master/mockup-images/mockup-seneca.png)

### __[mastermind.fyi](https://mastermind.fyi)__

# C. How Models were Trained

Each language model was fine-tuned with the [Hugging Face script](https://github.com/huggingface/transformers/tree/master/examples#language-model-training) and 3-4 training epochs were used for each model with the default script settings (took around 5-10 minutes on NVIDIA's Tesla P100 GPU, 16GB on GCP).

A single .txt training file (.csv also works) was prepared and cleaned for each mentor using various sources of text from books/articles they've written and speeches/interview on YouTube if "captions" with punctuation were available (you can use YouTube filter criteria to search for videos with "captions").

The amount of data used for each is incredibly small, yet the language generation model still does a remarkably good job of picking up the speaker's style in word choice/patterns.

##### *(Size of Training Data, # of Tokens)*

1. __Tony Robbins *(2 MB, 465K)*__
2. __Paul Graham *(1.1 MB, 240K)*__
2. __David Goggins *(616 KB, 156K)*__
3. __Brene Brown *(899 KB, 219K)*__
4. __Seneca *(1.3 MB, 334K)*__


# D. Deployment and Serving the App

__Hosting:__ The app was deployed using a docker image on a stand-alone VM from [Digital Ocean](https://cloud.digitalocean.com/) (2GB RAM, $10/month) and the domain (.fyi) was purchased from namecheap.com.  Each model is 550MB and hosted on Dropbox.  During the Docker build, the models (separate zipped folders) get downloaded (see 'server.py' script) and unzipped.  The .yaml file is not used with Digital Ocean, but works if you deploy the app with Google's App Engine.

__GPT-2 Generation Length:__ By default, the advice generation length is set to a max of 50 tokens in the "server.py" script and the generated text gets truncated down to end (typically) with the last punctuation (., ?, !) found before it reaches 50 tokens.  If the generated text does not contain ending punctuation, it automatically gets truncated to 100 characters (user-defined parameter in server.py script).

__Scale-ability:__ Only 1 concurrent request can be ran at a time.  Subsequently, 2 users cannot use the app at the same time (creates a "traffic queue") and this decision was made to control costs.  Google App Engine offers greater flexibility and load balancing, but comes at a higher cost.  Google Cloud Run is Google's stateless option that is cheaper; however, I experienced long delays (30-45 seconds minimum) to start/serve the site ("cold start problem") because of the size of the Docker image (about 1GB).

__How App is Ran:__ The app is built using the [Starlette](https://www.starlette.io/) app framework (lightweight, similar to Flask).  After clicking the "Generate Advice" button, the app loads the selected mentor's language model and for the chosen question, the language model uses 1 of 3 short "prompts/seeds" (selected at random) to generate the advice.

__Helpful Deployment Resources:__
1. [GPT-2: Google Cloud Run Example from Max Woolf](https://github.com/minimaxir/gpt-2-cloud-run) and [Tutorial](https://minimaxir.com/2019/09/howto-gpt2/)
2. [GPT-2: Basic Web App #1](https://github.com/NaxAlpha/gpt-2xy) and [Tutorial](https://medium.com/datadriveninvestor/deploy-machine-learning-model-in-google-cloud-using-cloud-run-6ced8ba52aac)
3. [GPT-2: Basic Web App #2](https://github.com/jingw222/gpt2-app)

__Inspiring Language Generation Demos:__
1. [Hugging Face Web Apps](https://transformer.huggingface.co/) 
2. [Writeup.ai](https://writeup.ai/) and [Tutorial/Lessons Learned](https://senrigan.io/blog/how-writeupai-runs-behind-the-scenes/)
3. ["News You CAN'T use"](http://newsyoucantuse.com/) from Adam Geitgey and [Tutorial](https://medium.com/@ageitgey/deepfaking-the-news-with-nlp-and-transformer-models-5e057ebd697d)
4. [Max Woolf's Interactive GPT-2 apps](https://minimaxir.com/portfolio/)
5. [Allen NLP GPT-2 app](https://demo.allennlp.org/next-token-lm?text=AllenNLP%20is)
6. [GPT-2 Poetry & Insights](https://www.gwern.net/GPT-2) from Gwern


# E. Web App Design

The front-end website was designed using the [Blocs](https://blocsapp.com/) app for the majority of the http/css and served with the Starlette app.

To optimize the css/javascript for faster web page loading, I used some insights from [Google's "PageSpeed Insights"](https://developers.google.com/speed/pagespeed/insights/) and leveraged [PurifyCSS](https://purifycss.online/) to remove unused css.  In Google Chrome, I also used [DevTools](https://developers.google.com/web/tools/lighthouse) to run the audit pages to find recommendations to improve user experience.

# F. Language Model (GPT-2) - How it Works

Here are some incredible tutorials on language models 👏 👏 👏

1. [The Annotated GPT-2](https://amaarora.github.io/2020/02/18/annotatedGPT2.html?utm_campaign=Data_Elixir&utm_source=Data_Elixir_273) 
2. [Jay Alammar's Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/)
3. [FloydHub's GPT-2 Overview](https://blog.floydhub.com/gpt2/)
4. [Allen NLP GPT-2 app](https://demo.allennlp.org/next-token-lm?text=AllenNLP%20is)

# G. Mastermind Principle Explained

Unlock your potential with the world class experts across disciplines!

The "mastermind principle" comes from Napoleon Hill (author of "Think and Grow Rich") and consists of an alliance of two or more minds working in perfect harmony for the attainment of a common definite objective.  Success does not come without the cooperation of others.

__[Try it out!](https://mastermind.fyi)__