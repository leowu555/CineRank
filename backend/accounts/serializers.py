from django.contrib.auth.models import User
from rest_framework import serializers

from ratings.models import Rating


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=6)

    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'password']

    def create(self, validated_data):
        return User.objects.create_user(
            username=validated_data['username'],
            email=validated_data.get('email', ''),
            password=validated_data['password'],
        )


class UserSerializer(serializers.ModelSerializer):
    ratings_count = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'date_joined', 'ratings_count']

    def get_ratings_count(self, user):
        return user.ratings.count()


class RatingSerializer(serializers.ModelSerializer):
    class Meta:
        model = Rating
        fields = [
            'id',
            'movie_id',
            'title',
            'poster_url',
            'score',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

    def validate(self, attrs):
        # Ownership always comes from the authenticated user, never the body.
        for key in ('user', 'user_id'):
            if key in getattr(self, 'initial_data', {}):
                raise serializers.ValidationError(
                    {key: 'Cannot set rating owner from the request body.'}
                )
        return attrs

    def validate_score(self, value):
        if value < 1 or value > 5:
            raise serializers.ValidationError('Score must be between 1 and 5.')
        return value
