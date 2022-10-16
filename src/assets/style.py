from dataclasses import dataclass


@dataclass
class TableStyle:
    width: str = 'auto'


@dataclass
class DataStyle:
    height: str = 'auto'
    whiteSpace: str = 'normal'
    lineHeight: str = '15px'
    textAlign: str = 'left'
    width: str = 'auto'


@dataclass
class CellStyle:
    overflow: str = 'hidden'
    textOverFlow: str = 'ellipsis'
    minWidth: str = '0px'
    textAlign: str = 'auto'
    width: str = 'auto'


@dataclass
class UploadStyle:
    width: str = 'auto',
    height: str = '60px',
    lineHeight: str = '60px',
    borderWidth: str = '1px',
    borderStyle: str = 'dashed',
    borderRadius: str = '5px',
    textAlign: str = 'center',
    margin: str = '10px'


@dataclass
class RadioItemsStyle:
    margin: str = '5px'
    display: str = 'block'


@dataclass
class ChecklistStyle:
    margin: str = '5px'
    display: str = 'block'
