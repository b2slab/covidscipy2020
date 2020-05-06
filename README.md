# covidscipy2020

Github for 2020 Scientific Python class, GPLv3

Goal: Develop a chatbot capable to recruit biomedical data from patients, including audio from coughing. Eventually deliver a probability of diagnosis. All software and material to be released publicly under open source GPLv3 license.

Structure. The project will be multifaceted, with modular elements built by teams. Main building blocs are:

  - Overall specification for features: Including recruiting of audio of coughing from patients, fever data, geolocalization.
    Database storage: Design and Use of a noSQL database to store data (e.g. Mongodb, https://www.mongodb.com/blog/post/getting-started-with-python-and-mongodb, https://www.w3schools.com/python/python_mongodb_getstarted.asp)
  - Chatbot operation with the user: Asynchronous scheme for interaction with users. Definition of state machine in the interaction (https://github.com/aiogram/aiogram, https://docs.python.org/3/library/asyncio.html)
  - Machine learning: Develop an algorithm from a set of audios capable of classification. The system can be tested with a set of sample audios made by the class emulating two kind of coughing (https://github.com/tyiannak/pyAudioAnalysis, https://github.com/jsingh811/pyAudioProcessing). Documentation, dissemination.
  - Creation of a dynamic website programmatically through python (using flask or django, flask is simpler). Parts of the website could connect to the db to show statistics of the system as a small dashboard. Creation of contents, code documentation (https://docs.python-guide.org/writing/documentation/, https://www.djangoproject.com/start/, https://flask.palletsprojects.com/, with flask you can integrate dash https://plotly.com/dash/) ) .
  - Privacy and ethics. Important part of project. Review legal framework, code a module which recruits and provide legal information and coverage of the project(https://gdpr-info.eu/, consultation with b2slab and CREB-UPC). 
