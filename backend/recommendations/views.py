from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from ratings.models import Rating
from recommendations.recsys.pipeline import recommend
from recommendations.recsys.types import RatingEvent


def _rating_events():
    rows = Rating.objects.values('user_id', 'movie_id', 'title', 'poster_url', 'score')
    return [
        RatingEvent(
            user_id=row['user_id'],
            item_id=row['movie_id'],
            score=row['score'],
            title=row['title'] or '',
            poster_url=row['poster_url'],
        )
        for row in rows
    ]


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def recommend_movies(request):
    result = recommend(_rating_events(), request.user.id)
    payload = {
        'results': [
            {
                'movie_id': item.item_id,
                'title': item.title,
                'poster_url': item.poster_url,
                'reason': item.reason,
            }
            for item in result.items
        ],
        'method': result.method,
    }
    if result.message:
        payload['message'] = result.message
    return Response(payload)
