"""Tests for symmetric only/exclude trimming semantics.

Serialization trims fields selected away by ``only``/``exclude`` from the
output. On deserialization, input values for those trimmed fields are
silently ignored (neither validated nor included in the result), even when
``unknown`` is ``RAISE``, unless the schema opts back into reporting them
with ``Meta.raise_on_trimmed = True``. ``dump_only`` fields are unaffected
and keep being reported as unknown fields.
"""

import pytest

from marshmallow import EXCLUDE, INCLUDE, RAISE, Schema, ValidationError, fields


class GrandChildSchema(Schema):
    goo = fields.Int()
    gah = fields.Int()


class ChildSchema(Schema):
    a = fields.Int()
    b = fields.Int()
    grand = fields.Nested(GrandChildSchema)


class ParentSchema(Schema):
    x = fields.Int()
    y = fields.Int()
    child = fields.Nested(ChildSchema)
    kids = fields.List(fields.Nested(ChildSchema))


@pytest.fixture
def parent_data():
    return {
        "x": 1,
        "y": 2,
        "child": {"a": 3, "b": 4, "grand": {"goo": 5, "gah": 6}},
        "kids": [{"a": 7, "b": 8}, {"a": 9, "b": 10}],
    }


class TestDumpTrimming:
    """Serialization side of the matrix: trimmed fields are dropped from output."""

    def test_dump_exclude_top_level(self, parent_data):
        result = ParentSchema(exclude=("y",)).dump(parent_data)
        assert "y" not in result
        assert result["x"] == 1

    def test_dump_only_top_level(self, parent_data):
        result = ParentSchema(only=("x",)).dump(parent_data)
        assert result == {"x": 1}

    def test_dump_exclude_dot_path(self, parent_data):
        result = ParentSchema(exclude=("child.b", "kids.b")).dump(parent_data)
        assert result["child"] == {"a": 3, "grand": {"goo": 5, "gah": 6}}
        assert result["kids"] == [{"a": 7}, {"a": 9}]

    def test_dump_only_multi_level_dot_path(self, parent_data):
        result = ParentSchema(only=("x", "child.grand.goo")).dump(parent_data)
        assert result == {"x": 1, "child": {"grand": {"goo": 5}}}


class TestLoadTrimmedIgnored:
    """Deserialization side: trimmed input fields are ignored, even under RAISE."""

    def test_load_exclude_top_level(self):
        result = ParentSchema(exclude=("y",)).load({"x": 1, "y": 2})
        assert result == {"x": 1}

    def test_load_only_top_level(self):
        result = ParentSchema(only=("x",)).load({"x": 1, "y": 2})
        assert result == {"x": 1}

    def test_load_exclude_dot_path(self):
        schema = ParentSchema(exclude=("child.b",))
        result = schema.load({"x": 1, "child": {"a": 3, "b": 4}})
        assert result == {"x": 1, "child": {"a": 3}}

    def test_load_only_dot_path(self):
        schema = ParentSchema(only=("x", "child.a"))
        result = schema.load({"x": 1, "y": 2, "child": {"a": 3, "b": 4}})
        assert result == {"x": 1, "child": {"a": 3}}

    def test_load_only_multi_level_dot_path(self):
        schema = ParentSchema(only=("x", "child.grand.goo"))
        result = schema.load({"x": 1, "child": {"a": 3, "grand": {"goo": 5, "gah": 6}}})
        assert result == {"x": 1, "child": {"grand": {"goo": 5}}}

    def test_load_exclude_multi_level_dot_path(self):
        schema = ParentSchema(exclude=("child.grand.gah",))
        result = schema.load({"x": 1, "child": {"grand": {"goo": 5, "gah": 6}}})
        assert result == {"x": 1, "child": {"grand": {"goo": 5}}}

    def test_load_nested_list_exclude_dot_path(self):
        schema = ParentSchema(exclude=("kids.b",))
        result = schema.load({"x": 1, "kids": [{"a": 7, "b": 8}, {"a": 9, "b": 10}]})
        assert result == {"x": 1, "kids": [{"a": 7}, {"a": 9}]}

    def test_load_nested_list_only_dot_path(self):
        schema = ParentSchema(only=("x", "kids.a"))
        result = schema.load({"x": 1, "kids": [{"a": 7, "b": 8}, {"a": 9, "b": 10}]})
        assert result == {"x": 1, "kids": [{"a": 7}, {"a": 9}]}

    def test_load_many(self):
        schema = ParentSchema(exclude=("y",), many=True)
        result = schema.load([{"x": 1, "y": 2}, {"x": 3}])
        assert result == [{"x": 1}, {"x": 3}]

    def test_trimmed_fields_are_not_validated(self):
        # Invalid values under trimmed keys do not raise
        schema = ParentSchema(exclude=("y", "child.b"))
        result = schema.load({"x": 1, "y": "not-an-int", "child": {"b": "junk"}})
        assert result == {"x": 1, "child": {}}

    def test_trimmed_field_with_data_key(self):
        class KeySchema(Schema):
            x = fields.Int(data_key="X")
            y = fields.Int(data_key="Y")

        result = KeySchema(exclude=("y",)).load({"X": 1, "Y": 2})
        assert result == {"x": 1}

    def test_undeclared_fields_still_raise(self):
        schema = ParentSchema(exclude=("y",))
        with pytest.raises(ValidationError) as excinfo:
            schema.load({"x": 1, "y": 2, "nope": 3})
        assert excinfo.value.messages == {"nope": ["Unknown field."]}


class TestLoadTrimmedUnknownModes:
    """Trimmed fields are ignored regardless of the unknown policy."""

    def test_unknown_raise_ignores_trimmed(self):
        schema = ParentSchema(exclude=("y",), unknown=RAISE)
        assert schema.load({"x": 1, "y": 2}) == {"x": 1}

    def test_unknown_exclude_ignores_trimmed(self):
        schema = ParentSchema(exclude=("y",), unknown=EXCLUDE)
        assert schema.load({"x": 1, "y": 2}) == {"x": 1}

    def test_unknown_include_does_not_include_trimmed(self):
        schema = ParentSchema(exclude=("y",), unknown=INCLUDE)
        result = schema.load({"x": 1, "y": 2, "z": 3})
        # Trimmed field is ignored; genuinely unknown fields are included
        assert result == {"x": 1, "z": 3}

    def test_unknown_include_nested(self):
        schema = ParentSchema(only=("x", "child.a"))
        result = schema.load({"x": 1, "child": {"a": 3, "b": 4}}, unknown=INCLUDE)
        assert result == {"x": 1, "child": {"a": 3}}


class TestRaiseOnTrimmedOptOut:
    """Meta.raise_on_trimmed = True falls back to reporting trimmed fields."""

    def make_schema(self, **kwargs):
        class StrictSchema(ParentSchema):
            class Meta:
                raise_on_trimmed = True

        return StrictSchema(**kwargs)

    def test_exclude_top_level_raises(self):
        schema = self.make_schema(exclude=("y",))
        with pytest.raises(ValidationError) as excinfo:
            schema.load({"x": 1, "y": 2})
        assert excinfo.value.messages == {"y": ["Unknown field."]}

    def test_only_top_level_raises(self):
        schema = self.make_schema(only=("x",))
        with pytest.raises(ValidationError) as excinfo:
            schema.load({"x": 1, "y": 2})
        assert excinfo.value.messages == {"y": ["Unknown field."]}

    def test_exclude_dot_path_raises(self):
        # The opt-out is schema-level: the nested schema must set it itself
        class StrictChildSchema(Schema):
            a = fields.Int()
            b = fields.Int()

            class Meta:
                raise_on_trimmed = True

        class StrictParentSchema(Schema):
            x = fields.Int()
            child = fields.Nested(StrictChildSchema)

        schema = StrictParentSchema(exclude=("child.b",))
        with pytest.raises(ValidationError) as excinfo:
            schema.load({"x": 1, "child": {"a": 3, "b": 4}})
        assert excinfo.value.messages == {"child": {"b": ["Unknown field."]}}

    def test_nested_list_dot_path_raises(self):
        class StrictChildSchema(Schema):
            a = fields.Int()
            b = fields.Int()

            class Meta:
                raise_on_trimmed = True

        class StrictParentSchema(Schema):
            x = fields.Int()
            kids = fields.List(fields.Nested(StrictChildSchema))

        schema = StrictParentSchema(exclude=("kids.b",))
        with pytest.raises(ValidationError) as excinfo:
            schema.load({"x": 1, "kids": [{"a": 7, "b": 8}]})
        assert excinfo.value.messages == {"kids": {0: {"b": ["Unknown field."]}}}

    def test_unknown_exclude_still_ignores(self):
        # EXCLUDE drops all unknown keys, trimmed or not
        schema = self.make_schema(exclude=("y",))
        assert schema.load({"x": 1, "y": 2}, unknown=EXCLUDE) == {"x": 1}

    def test_unknown_include_includes_trimmed(self):
        # Opted-out trimmed fields behave like ordinary unknown fields
        schema = self.make_schema(exclude=("y",))
        result = schema.load({"x": 1, "y": 2}, unknown=INCLUDE)
        assert result == {"x": 1, "y": 2}

    def test_dump_unaffected_by_opt_out(self, parent_data):
        schema = self.make_schema(exclude=("y",))
        result = schema.dump(parent_data)
        assert "y" not in result


class TestDumpOnlyUnaffected:
    """dump_only fields in input keep being reported as unknown fields."""

    def make_schema(self, **kwargs):
        class DumpOnlySchema(Schema):
            x = fields.Int()
            d = fields.Int(dump_only=True)

        return DumpOnlySchema(**kwargs)

    def test_dump_only_raises_by_default(self):
        with pytest.raises(ValidationError) as excinfo:
            self.make_schema().load({"x": 1, "d": 2})
        assert excinfo.value.messages == {"d": ["Unknown field."]}

    def test_dump_only_raises_with_raise_on_trimmed(self):
        class StrictDumpOnlySchema(Schema):
            d = fields.Int(dump_only=True)

            class Meta:
                raise_on_trimmed = True

        with pytest.raises(ValidationError) as excinfo:
            StrictDumpOnlySchema().load({"d": 2})
        assert excinfo.value.messages == {"d": ["Unknown field."]}

    def test_dump_only_excluded_also_raises(self):
        # A field that is both dump_only and excluded is still reported
        with pytest.raises(ValidationError) as excinfo:
            self.make_schema(exclude=("d",)).load({"x": 1, "d": 2})
        assert excinfo.value.messages == {"d": ["Unknown field."]}

    def test_dump_only_unknown_exclude(self):
        result = self.make_schema().load({"x": 1, "d": 2}, unknown=EXCLUDE)
        assert result == {"x": 1}
