This dataset, named Twibot-20, is a comprehensive sample of the Twittersphere.

This dataset focuses on bot detection on Twitter.

The datasets are released under MIT(https://github.com/GabrielHam/TwiBot-20/blob/main/LICENSE)

The train, validation, test and support set are provided in .json format.

The user in support set provides the neighborhood information.

Each user sample contains:
- 'ID': the ID from Twitter identifying the user.
- 'profile': the profile information obtained from Twitter API.
- 'tweet': the recent 200 tweets of this user.
- 'neighbor': the random 20 followers and followings of this user.
- 'domain': the domain of this user and the domains include politics, business, entertainment and sports.
- 'label': the label of this user and '1' means it is a bot while '0' means it is a human.