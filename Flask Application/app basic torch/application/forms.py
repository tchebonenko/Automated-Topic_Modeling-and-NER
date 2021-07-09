from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import SelectField, TextField, SubmitField
from wtforms.widgets import TextArea
from wtforms.validators import DataRequired


class TextForm(FlaskForm):
    text = TextField('Text', validators=[DataRequired()], widget=TextArea())
    submit = SubmitField('Process Consultation Request')
