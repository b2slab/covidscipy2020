# Enable reads (GET), inserts (POST) and DELETE for resources/collections
# (if you omit this line, the API will default to ['GET'] and provide
# read-only access to the endpoint).
RESOURCE_METHODS = ['GET', 'POST']

# Enable reads (GET), edits (PATCH), replacements (PUT) and deletes of
# individual items  (defaults to read-only item access).
ITEM_METHODS = ['GET', 'PATCH', 'PUT', 'DELETE']


MONGO_HOST = 'localhost'
MONGO_PORT = 27017
MONGO_DBNAME = 'CovidDatabase'

#Other settings
PAGINATION = False


dataSchema = {
	"username": {
		'type': 'string',
		'required': True,
		'unique': True,
		'minlength': 1,
		'maxlength': 16
	},

	"age": {
		'type': 'integer',
		'required': True
	},

	"body_temp": {
		'type': 'number',
		'required': True
	},

	"location": {
        'type': 'dict',
		'required': True,
        'schema': {
            'country': {
				'type': 'string',
				'required': True,
				'allowed': ["Spain", "France", "Sweden"]
			},
            'city': {
				'type': 'string',
				'required': False,
				'allowed': ["Barcelona", "Madrid"]
			}
        },
    },

	"audio_features": {
		'type': 'dict',
		'required': True,
		'schema': {
			"chroma_10_mean": {
				'type': 'string'
			},
			"chroma_10_std": {
				'type': 'string'
			},
			"chroma_11_mean": {
				'type': 'string'
			},
			"chroma_11_std": {
				'type': 'string'
			},
			"chroma_12_mean": {
				'type': 'string'
			},
			"chroma_12_std": {
				'type': 'string'
			},
			"chroma_1_mean": {
				'type': 'string'
			},
			"chroma_1_std": {
				'type': 'string'
			},
			"chroma_2_mean": {
				'type': 'string'
			},
			"chroma_2_std": {
				'type': 'string'
			},
			"chroma_3_mean": {
				'type': 'string'
			},
			"chroma_3_std": {
				'type': 'string'
			},
			"chroma_4_mean": {
				'type': 'string'
			},
			"chroma_4_std": {
				'type': 'string'
			},
			"chroma_5_mean": {
				'type': 'string'
			},
			"chroma_5_std": {
				'type': 'string'
			},
			"chroma_6_mean": {
				'type': 'string'
			},
			"chroma_6_std": {
				'type': 'string'
			},
			"chroma_7_mean": {
				'type': 'string'
			},
			"chroma_7_std": {
				'type': 'string'
			},
			"chroma_8_mean": {
				'type': 'string'
			},
			"chroma_8_std": {
				'type': 'string'
			},
			"chroma_9_mean": {
				'type': 'string'
			},
			"chroma_9_std": {
				'type': 'string'
			},
			"chroma_std_mean": {
				'type': 'string'
			},
			"chroma_std_std": {
				'type': 'string'
			},
			"delta chroma_10_mean": {
				'type': 'string'
			},
			"delta chroma_10_std": {
				'type': 'string'
			},
			"delta chroma_11_mean": {
				'type': 'string'
			},
			"delta chroma_11_std": {
				'type': 'string'
			},
			"delta chroma_12_mean": {
				'type': 'string'
			},
			"delta chroma_12_std": {
				'type': 'string'
			},
			"delta chroma_1_mean": {
				'type': 'string'
			},
			"delta chroma_1_std": {
				'type': 'string'
			},
			"delta chroma_2_mean": {
				'type': 'string'
			},
			"delta chroma_2_std": {
				'type': 'string'
			},
			"delta chroma_3_mean": {
				'type': 'string'
			},
			"delta chroma_3_std": {
				'type': 'string'
			},
			"delta chroma_4_mean": {
				'type': 'string'
			},
			"delta chroma_4_std": {
				'type': 'string'
			},
			"delta chroma_5_mean": {
				'type': 'string'
			},
			"delta chroma_5_std": {
				'type': 'string'
			},
			"delta chroma_6_mean": {
				'type': 'string'
			},
			"delta chroma_6_std": {
				'type': 'string'
			},
			"delta chroma_7_mean": {
				'type': 'string'
			},
			"delta chroma_7_std": {
				'type': 'string'
			},
			"delta chroma_8_mean": {
				'type': 'string'
			},
			"delta chroma_8_std": {
				'type': 'string'
			},
			"delta chroma_9_mean": {
				'type': 'string'
			},
			"delta chroma_9_std": {
				'type': 'string'
			},
			"delta chroma_std_mean": {
				'type': 'string'
			},
			"delta chroma_std_std": {
				'type': 'string'
			},
			"delta energy_entropy_mean": {
				'type': 'string'
			},
			"delta energy_entropy_std": {
				'type': 'string'
			},
			"delta energy_mean": {
				'type': 'string'
			},
			"delta energy_std": {
				'type': 'string'
			},
			"delta mfcc_10_mean": {
				'type': 'string'
			},
			"delta mfcc_10_std": {
				'type': 'string'
			},
			"delta mfcc_11_mean": {
				'type': 'string'
			},
			"delta mfcc_11_std": {
				'type': 'string'
			},
			"delta mfcc_12_mean": {
				'type': 'string'
			},
			"delta mfcc_12_std": {
				'type': 'string'
			},
			"delta mfcc_13_mean": {
				'type': 'string'
			},
			"delta mfcc_13_std": {
				'type': 'string'
			},
			"delta mfcc_1_mean": {
				'type': 'string'
			},
			"delta mfcc_1_std": {
				'type': 'string'
			},
			"delta mfcc_2_mean": {
				'type': 'string'
			},
			"delta mfcc_2_std": {
				'type': 'string'
			},
			"delta mfcc_3_mean": {
				'type': 'string'
			},
			"delta mfcc_3_std": {
				'type': 'string'
			},
			"delta mfcc_4_mean": {
				'type': 'string'
			},
			"delta mfcc_4_std": {
				'type': 'string'
			},
			"delta mfcc_5_mean": {
				'type': 'string'
			},
			"delta mfcc_5_std": {
				'type': 'string'
			},
			"delta mfcc_6_mean": {
				'type': 'string'
			},
			"delta mfcc_6_std": {
				'type': 'string'
			},
			"delta mfcc_7_mean": {
				'type': 'string'
			},
			"delta mfcc_7_std": {
				'type': 'string'
			},
			"delta mfcc_8_mean": {
				'type': 'string'
			},
			"delta mfcc_8_std": {
				'type': 'string'
			},
			"delta mfcc_9_mean": {
				'type': 'string'
			},
			"delta mfcc_9_std": {
				'type': 'string'
			},
			"delta spectral_centroid_mean": {
				'type': 'string'
			},
			"delta spectral_centroid_std": {
				'type': 'string'
			},
			"delta spectral_entropy_mean": {
				'type': 'string'
			},
			"delta spectral_entropy_std": {
				'type': 'string'
			},
			"delta spectral_flux_mean": {
				'type': 'string'
			},
			"delta spectral_flux_std": {
				'type': 'string'
			},
			"delta spectral_rolloff_mean": {
				'type': 'string'
			},
			"delta spectral_rolloff_std": {
				'type': 'string'
			},
			"delta spectral_spread_mean": {
				'type': 'string'
			},
			"delta spectral_spread_std": {
				'type': 'string'
			},
			"delta zcr_mean": {
				'type': 'string'
			},
			"delta zcr_std": {
				'type': 'string'
			},
			"energy_entropy_mean": {
				'type': 'string'
			},
			"energy_entropy_std": {
				'type': 'string'
			},
			"energy_mean": {
				'type': 'string'
			},
			"energy_std": {
				'type': 'string'
			},
			"label": {
				'type': 'string'
			},
			"mfcc_10_mean": {
				'type': 'string'
			},
			"mfcc_10_std": {
				'type': 'string'
			},
			"mfcc_11_mean": {
				'type': 'string'
			},
			"mfcc_11_std": {
				'type': 'string'
			},
			"mfcc_12_mean": {
				'type': 'string'
			},
			"mfcc_12_std": {
				'type': 'string'
			},
			"mfcc_13_mean": {
				'type': 'string'
			},
			"mfcc_13_std": {
				'type': 'string'
			},
			"mfcc_1_mean": {
				'type': 'string'
			},
			"mfcc_1_std": {
				'type': 'string'
			},
			"mfcc_2_mean": {
				'type': 'string'
			},
			"mfcc_2_std": {
				'type': 'string'
			},
			"mfcc_3_mean": {
				'type': 'string'
			},
			"mfcc_3_std": {
				'type': 'string'
			},
			"mfcc_4_mean": {
				'type': 'string'
			},
			"mfcc_4_std": {
				'type': 'string'
			},
			"mfcc_5_mean": {
				'type': 'string'
			},
			"mfcc_5_std": {
				'type': 'string'
			},
			"mfcc_6_mean": {
				'type': 'string'
			},
			"mfcc_6_std": {
				'type': 'string'
			},
			"mfcc_7_mean": {
				'type': 'string'
			},
			"mfcc_7_std": {
				'type': 'string'
			},
			"mfcc_8_mean": {
				'type': 'string'
			},
			"mfcc_8_std": {
				'type': 'string'
			},
			"mfcc_9_mean": {
				'type': 'string'
			},
			"mfcc_9_std": {
				'type': 'string'
			},
			"spectral_centroid_mean": {
				'type': 'string'
			},
			"spectral_centroid_std": {
				'type': 'string'
			},
			"spectral_entropy_mean": {
				'type': 'string'
			},
			"spectral_entropy_std": {
				'type': 'string'
			},
			"spectral_flux_mean": {
				'type': 'string'
			},
			"spectral_flux_std": {
				'type': 'string'
			},
			"spectral_rolloff_mean": {
				'type': 'string'
			},
			"spectral_rolloff_std": {
				'type': 'string'
			},
			"spectral_spread_mean": {
				'type': 'string'
			},
			"spectral_spread_std": {
				'type': 'string'
			},
			"zcr_mean": {
				'type': 'string'
			},
			"zcr_std": {
				'type': 'string'
			}
		}
	},

	"audio_file": {
		'type': 'media',
		'required': False
	}
}

dataEndpoint = {
    # 'title' tag used in item links. Defaults to the resource title minus
    # the final, plural 's' (works fine in most cases but not for 'people')
    'item_title': 'userData',

    # by default the standard item entry point is defined as
    # '/data/<ObjectId>'. We can add an additional endpoint.
	# This way consumers can also perform
    # GET requests at '/data/<user_id>'.
    'additional_lookup': {
        'url': 'regex("[\w]+")',
        'field': 'username'
    },


    # most global settings can be overridden at resource level
    #'resource_methods': ['GET', 'POST'],

    'schema': dataSchema
}


DOMAIN = {
	'data': dataEndpoint
}
