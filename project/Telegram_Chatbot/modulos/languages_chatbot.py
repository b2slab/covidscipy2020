def import_languages():
    questions = {
        "en":{
            "q1":"Hi there! Please, enter your name.",
            "q2":"Cancelled",
            "q3":"Allright we'll stop here...\n\n Please, give me a second while I upload the data.",
            "q4":"That's it!",
            "q5":"Process stopped",
            "q6":"Username is invalid. Please enter a correct username.",
            "q7":"How old are you?",
            "q8":"Age has to be a number.\n\n How old are you? (digits only)",
            "q9":"Male",
            "q10":"Female",
            "q11":"Other",
            "q12":"Inadequate answer. Please choose one of the provided options",
            "q13":"Send Current Location",
            "q14":"Would you mind to send us your current location?\n\nPlease activate location on your device.",
            "q15":"Please activate location on your device and send it to us.",
            "q16":"Currently positive",
            "q17":"Had covid in the past",
            "q18":"Never been diagnosed",
            "q19":"Do you have Covid-19, have had it in the past, or have never been diagnosed?",
            "q20":"Could you send us a recording of your cough?",
            "q21":"Just use the audio message option from telegram and cough to the microphone.",
            "q22":"Please, give me a second while I analyze you cough...",
            "q23":"Sorry, we didn't recognize this as cough. Please, cough again",
            "q24":"Thanks for your cough",
            "q25":"Do you have dry cough?",
            "q26":"Yes",
            "q27":"No",
            "q28":"Do you smoke regularly?",
            "q29":"Do you suffer a cold?",
            "q30":"Do find it difficult to breathe?",
            "q31":"Do you have a sore throat?",
            "q32":"Do you have chest pain or pressure?",
            "q33":"Do you have difficulty breathing or shortness of breath?",
            "q34":"Do you have any other information you would like to add?",
            "q35":"Thank you very much for you collaboration!\n\n Please, give me a second while I upload the data.",

            "q36":"What is your gender?",

            "q37":"Hello, you are chatting with CovidScipy2020's bot.\
            \n\nThis bot is designed to gather data about people who may,\
            or may not have Sars-covid-2019, in order to better understand the disease\
            and potentially help you to know if you may be susceptible to have the virus,\
            just by providing us your symptoms.\
            \n\nRight now it is only in a data-gathering state, so you would help us a lot\
            by just adding your information (or someone else's if you have their permission).\
            \n\nYou can access and delete your data at anyime.",

            "q38":"Okay. You may now add data and symptoms of your own, or from someone else you are responsible of.\
            \n\nWe will begin with your first name, just to identify you in case you add data from your relatives.\
            \n\n\nWhat is your name?\
            \n\n\n(Use the command /cancel at any time to go back to the menu. No entry will be uploaded)",

            "q39":"Have you been (fully) vaccinated?",

            "q40":"Do you have fever?",

            "q41":"Do you feel more tired than usual?",

            "q42":"Do you feel muscular pain?",

            "q43":"Do you have a diminished sense of smell/taste?",

            "q44":"Do you suffer pneumonia?",

            "q45":"Do you have diarrhea?",

            "q46":"Do you have hypertension?",

            "q47":"Do you have asthma?",

            "q48":"Do you have diabetes?",

            "q49":"Do you have a Chronic Lung Disease?",

            "q50":"Do you have Ischemic Heart Disease?",

            "q51":'The audio recording is too short.\n\nPlease repeat the recording.',

            "q52":'The audio recording is too long.\n\nPlease repeat the recording.',

            "q53":'According to our models, you are POSITIVE in COVID-19',

            "q54":'According to our models, you are NEGATIVE in COVID-19',

            "q55":"This is the end of the form. Do you want to add any extra information?",

            "q56":"Add data",

            "q57":"Delete data",

            "q58":"About",

            "q59":"Exit",

            "q60":"Welcome to covid scipy %s. Your id is %s. Select one of the following",

            "q61":"CANCEL",

            "q62":"These are the entries you have uploaded. Which one do you want to delete?",

            "q63":"%s. Do you want to delete more entries?",

            "q64":"Bye!",

            "q65":"Please write it here:",

            "q66":"Skip",

            "q67": "No entries found"



        },
        "es":{
            "q1":"¡Hola!\n\nPor favor, introduzca su nombre.",
            "q2":"Cancelado",
            "q3":"El proceso de recopilación de datos ha sido cancelado.",
            "q4":"¡Sus datos han sido almacenados correctamente!",
            "q5":"Proceso detenido",
            "q6":"Usuario inválido. Por favor, introduzca un nombre correcto (sólo texto).",
            "q7":"¿Cuántos años tiene?",
            "q8":"Su edad debe ser un número.\n\n¿Cuántos años tiene?",
            "q9":"Hombre",
            "q10":"Mujer",
            "q11":"Otro",
            "q12":"Respuesta incorrecta. Por favor, elija una de las opciones del teclado",
            "q13":"Enviar su ubicación",
            "q14":"¿Le importaría enviarnos su ubicación?\n\n Por favor, active la ubicación en su dispositivo.",
            "q15":"Por favor, active la ubicación en su dispositivo y envíenosla",
            "q16":"Positivo",
            "q17":"Negativo",
            "q18":"Desconocido",
            "q19":"¿Tiene usted COVID-19?",
            "q20":"¿Podría enviarnos una grabación de audio de su tos?",
            "q21":"Por favor, envíenos una nota de voz de su tos.",
            "q22":"Analizando su audio... (si tarda más de 15 segundos en responder, vuelva a enviarlo por favor)",
            "q23":"Lo sentimos, no hemos reconocido su audio como tos.\n\n¿Podría volver a intentarlo?",
            "q24":"Su audio ha sido aceptado. Muchas gracias por su tos.",
            "q25":"¿Padece usted tos seca?",
            "q26":"Sí",
            "q27":"No",
            "q28":"¿Es usted fumador/a?",
            "q29":"¿Esta usted resfriado/a?",
            "q30":"¿Padece usted dificultadades respiratorias?",
            "q31":"¿Tiene la garganta seca?",
            "q32":"¿Padece dolor o presión en el pecho?",
            "q33":"¿Tiene dificultades para respirar?",
            "q34":"¿Hay algo más que quiera añadir?",
            "q35":"¡Muchas gracias por su colaboración!\n\n Aguarde un segundo mientras se guardan los datos.",

            "q36":"¿Cuál es su género?",

            "q37":"Hola, usted está hablando con el bot llamado CovidScipy2020.\
            \n\nÉste bot está diseñado para recopilar datos sobre personas y su estado de salud respecto al SARS-COVID-19.\
            El objetivo del proyecto es comprender mejor la enfermedad y ayudar a los usuarios a saber si son susceptibles de tener el virus. Para hacerlo, le pediremos si nos puede facilitar sus síntomas y una muestra de su tos.\
            \n\nActualmente, el proyecto aún está en una fase preliminar de almacenamiento de datos. Por lo tanto, si nos pudiera facilitar sus datos nos sería de una tremenda utilidad (o los de otra persona a su cargo y con su consentimiento).\
            \n\nPor supuesto, usted puede acceder o eliminar la información proporcionada en cualquier momento.",

            "q38":"A continuación puede agregar datos y síntomas propios. También puede optar por facilitarnos los datos de una persona a su cargo.\
            \n\nComenzaremos por su nombre. El nombre que nos proporcione servirá como identificador en el caso de que quiera agregar los datos de sus familiares.\
            \n\n\n¿Cómo se llama usted?\
            \n\n\n(Escriba el comando /cancel en cualquier momento para volver al menú. Ningún dato será almacenado en tal caso.)",

            "q39":"¿Ha sido (completamente) vacunado/a?",

            "q40":"¿Tiene fiebre?",

            "q41":"¿Se encuentra más cansado/a de lo habitual?",

            "q42":"¿Siente algún tipo de dolor muscular?",

            "q43":"¿Está experimentando una pérdida del sentido del olfato o gusto?",

            "q44":"¿Alguna vez ha sufrido pulmonía?",

            "q45":"¿Tiene diarrea?",

            "q46":"¿Tiene hipertensión?",

            "q47":"¿Tiene asma?",

            "q48":"¿Tiene diabetes de algún tipo?",

            "q49":"¿Sufre algún tipo de enfermedad pulmonar crónica?",

            "q50":"¿Sufre la enfermedad de las arterias coronarias?",

            "q51":'La grabación de audio es demasiado corta.\n\nPor favor, repita la grabación.',

            "q52":'La grabación de audio es demasiado larga.\n\nPor favor, repita la grabación.',

            "q53":'Según nuestros modelos, usted es susceptible a ser POSITIVO en COVID-19.',

            "q54":'Según nuestros modelos, usted es NEGATIVO en COVID-19.',

            "q55":"Este es el final del formulario. ¿Desea agregar más información?",

            "q56":"Añadir datos",

            "q57":"Eliminar datos",

            "q58":"Acerca de",

            "q59":"Salir",

            "q60":"Bienvenido/a al covid scipy %s. Su id es %s. Seleccione una de las opciones siguientes:",

            "q61":"CANCELAR",

            "q62":"Estos son los registros que ha creado. ¿Cuál de ellos desea eliminar?",

            "q63":"%s. ¿Desea eliminar más registros?",

            "q64":"¡Hasta luego! ¡Gracias!",

            "q65":"Por favor, escríbalo aquí:",

            "q66":"Omitir",

            "q67":"No se han econtrado entradas"

        },
        "ca":{
            "q1":"Hola!\n\nSi us plau, introdueixi su nombre.",
            "q2":"Cancel·lat",
            "q3":"El procés de recopilació de dades ha estat cancel·lat.",
            "q4":"Les seves dades han estat emmagatzemades correctament!",
            "q5":"Procés detingut",
            "q6":"Usuari invàlid. Si us plau, introdueixi un nom correcte (només text).",
            "q7":"Quants anys té?",
            "q8":"La seva edat ha de ser un nombre.\n\nQuants anys té?",
            "q9":"Home",
            "q10":"Dona",
            "q11":"Altre",
            "q12":"Resposta incorrecta. Si us plau, triï una de les opcions del teclat",
            "q13":"Enviar ubicació",
            "q14":"Li faria res enviar-nos la seva ubicació?\n\nSi us plau, activi la ubicació al seu dispositiu.",
            "q15":"Si us plau, activi la ubicació en el seu dispositiu i envieu-nos-la",
            "q16":"Positiu",
            "q17":"Negatiu",
            "q18":"Desconegut",
            "q19":"Té vostè COVID-19?",
            "q20":"Podria enviar-nos un enregistrament d'àudio del seu estossec?",
            "q21":"Si us plau, envieu-nos una nota de veu del seu estossec.",
            "q22":"Analitzant el seu àudio...",
            "q23":"Ho sentim, no hem reconegut el seu àudio com a estossec.\n\n¿Podria tornar-ho a intentar?",
            "q24":"El seu àudio ha estat acceptat. Moltes gràcies pel seu estossec.",
            "q25":"Pateix vostè tos seca?",
            "q26":"Sí",
            "q27":"No",
            "q28":"És vostè fumador/a?",
            "q29":"Està vostè refredat/da",
            "q30":"Pateix vostè dificultats respiratòries?",
            "q31":"Té la gola seca?",
            "q32":"Pateix dolor o pressió al pit?",
            "q33":"Té dificultats per respirar?",
            "q34":"Hi ha alguna cosa més que vulgui afegir?",
            "q35":"Moltes gràcies per la seva col·laboració!\n\nEsperi un segon mentre s'emmagatzemen les dades.",

            "q36":"Quin és el seu gènere?",

            "q37":"Hola, vostè està parlant amb el bot anomenat CovidScipy2020.\
            \n\nAquest bot està dissenyat per recopilar dades sobre persones i el seu estat de salut respecte a la SARS-COVID-19.\
            L'objectiu del projecte és comprendre millor la malaltia i ajudar els usuaris a saber si són susceptibles de tenir el virus. Per fer-ho, li demanarem si ens pot facilitar els seus símptomes i una mostra d'àudio del seu estossec.\
            \n\nActualment, el projecte encara es troba en una fase preliminar d'emmagatzematge de dades. Per tant, si ens pogués facilitar les seves dades ens seria de molta utilitat (o les d'una altra persona al seu càrrec i amb el seu consentiment). \
            \n\nPer descomptat, vostè pot accedir o eliminar la informació proporcionada en qualsevol moment.",

            "q38":"A continuació pot afegir les dades i els símptomes propis. També pot optar per facilitar-nos les dades d'una persona al seu càrrec.\
            \n\nComençarem pel seu nom. El nom que ens proporcioni servirà com a identificador en el cas que vulgui afegir les dades dels seus familiars.\
            \n\n\nCom es diu vostè?\
            \n\n\n (Escrigui la comanda /cancel en qualsevol moment per a tornar al menú inicial. Cap dada serà emmagatzemat si ho fa)",

            "q39":"Ha estat (completament) vacunat/da?",

            "q40":"Té febre?",

            "q41":"Es troba més cansat/da del que és habitual?",

            "q42":"Sent algun tipus de dolor muscular?",

            "q43":"Està experimentant una pèrdua de el sentit de l'olfacte o gust?",

            "q44":"Alguna vegada ha patit pulmonia?",

            "q45":"Té diarrea?",

            "q46":"Té hipertensió?",

            "q47":"Té asma?",

            "q48":"Té diabetis d'algun tipus?",

            "q49":"Pateix algun tipus de malaltia pulmonar crònica?",

            "q50":"Pateix la malaltia de les artèries coronàries?",

            "q51":"L'enregistrament d'àudio és massa curt.\n\nSi us plau, repeteixi l'enregistrament.",

            "q52":"L'enregistrament d'àudio és massa llarg.\n\nSi us plau, repeteixi l'enregistrament.",

            "q53":"Segons els nostres models, vostè és susceptible a ser POSITIU en COVID-19.",

            "q54":"Segons els nostres models, vostè és NEGATIU a COVID-19.",

            "q55":"Aquest és el final del formulari. Voleu afegir més informació?",

            "q56":"Afegir dades",

            "q57":"Eliminar dades",

            "q58":"Sobre nosaltres",

            "q59":"Sortir",

            "q60":"Benvingut/da al covid scipy %s. La seva id és %s. Seleccioni una de les opcions següents:",

            "q61":"CANCELAR",

            "q62":"Aquests són els registres que ha creat. Quin d'ells desitja eliminar?",

            "q63":"%s. Vol eliminar més registres?",

            "q64":"¡Fins després! ¡Gràcies!",

            "q65":"Si us plau, escrigui-ho aquí:",

            "q66":"Ometre",

            "q67": "No s'han trobat entrades"

        }
    }
    return questions
