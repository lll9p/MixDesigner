# MixDesigner

Mixture experiment design helper

# Usage

    import models
    model = models.SimplexCentroid(point=5, lower_bounds=[0.6,0,0,0,0])
    model.fit(y)
    model.predict(X)
    model.score(X,y)
